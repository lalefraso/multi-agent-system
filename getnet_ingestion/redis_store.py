from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from redis import Redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import RedisError, ResponseError

from getnet_ingestion.embedder import EMBEDDING_DIMENSIONS, MODEL_NAME


DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_LOCALES = ("en", "mx")
ALL_LOCALES = (*SUPPORTED_LOCALES, "all")
DEFAULT_INPUT = DATA_DIR / "getnet_en_embeddings.jsonl"
DEFAULT_INDEX = "idx:getnet"
DEFAULT_PREFIX = "getnet:section:"
LEGACY_INDEXES = ("idx:getnet:en", "idx:getnet:mx")
PIPELINE_BATCH_SIZE = 200
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

LOGGER = logging.getLogger(__name__)


class RedisStoreError(RuntimeError):
    """Raised when embedding records cannot be validated or stored."""


def _validate_locale(locale: str) -> str:
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale {locale!r}; expected one of {SUPPORTED_LOCALES}")
    return locale


def embeddings_for_locale(locale: str) -> Path:
    return DATA_DIR / f"getnet_{_validate_locale(locale)}_embeddings.jsonl"


def load_embedding_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RedisStoreError(
                        f"Invalid JSON on line {line_number} of {input_path}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise RedisStoreError(
                        f"Expected a JSON object on line {line_number} of {input_path}"
                    )

                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise RedisStoreError(
                        f"Missing string id on line {line_number} of {input_path}"
                    )
                if record_id in seen_ids:
                    raise RedisStoreError(
                        f"Duplicate record id {record_id!r} in {input_path}"
                    )
                seen_ids.add(record_id)
                _validate_embedding(record, line_number, input_path)
                records.append(record)
    except OSError as exc:
        raise RedisStoreError(f"Could not read embeddings {input_path}: {exc}") from exc

    if not records:
        raise RedisStoreError(f"Embeddings file {input_path} contains no records")
    return records


def _validate_embedding(record: dict[str, Any], line_number: int, input_path: Path) -> None:
    record_id = record["id"]
    embedding = record.get("embedding")
    if record.get("embedding_model") != MODEL_NAME:
        raise RedisStoreError(
            f"Record {record_id!r} on line {line_number} of {input_path} "
            f"must use {MODEL_NAME}"
        )
    if record.get("embedding_dimensions") != EMBEDDING_DIMENSIONS:
        raise RedisStoreError(
            f"Record {record_id!r} on line {line_number} of {input_path} "
            f"must declare {EMBEDDING_DIMENSIONS} dimensions"
        )
    if (
        not isinstance(embedding, list)
        or len(embedding) != EMBEDDING_DIMENSIONS
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in embedding
        )
    ):
        raise RedisStoreError(
            f"Record {record_id!r} on line {line_number} of {input_path} "
            f"must contain {EMBEDDING_DIMENSIONS} finite numeric values"
        )
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise RedisStoreError(
            f"Record {record_id!r} on line {line_number} of {input_path} "
            "must contain non-empty text"
        )


def _ensure_index(client: Redis, index_name: str, prefix: str) -> None:
    index = client.ft(index_name)
    try:
        index.info()
        return
    except ResponseError as exc:
        error_message = str(exc).lower()
        missing_index_errors = (
            "unknown index name",
            "search_index_not_found",
            "index not found",
        )
        if not any(marker in error_message for marker in missing_index_errors):
            raise

    index.create_index(
        [
            TagField("category"),
            TagField("language"),
            TextField("page_title"),
            TextField("section_heading"),
            TextField("text"),
            TagField("embedding_model"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": EMBEDDING_DIMENSIONS,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        ],
        definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
    )


def _drop_index(client: Redis, index_name: str) -> None:
    index = client.ft(index_name)
    try:
        index.info()
    except ResponseError as exc:
        error_message = str(exc).lower()
        missing_index_errors = (
            "unknown index name",
            "search_index_not_found",
            "index not found",
        )
        if any(marker in error_message for marker in missing_index_errors):
            return
        raise
    index.dropindex(delete_documents=True)


def _to_hash(record: dict[str, Any]) -> dict[str, str | bytes]:
    hash_record: dict[str, str | bytes] = {}
    for field in ("id", "url", "page_title", "section_heading", "category", "language", "text"):
        value = record.get(field)
        if isinstance(value, str):
            hash_record[field] = value
    hash_record["embedding_model"] = MODEL_NAME
    hash_record["embedding_dimensions"] = str(EMBEDDING_DIMENSIONS)
    hash_record["embedding"] = struct.pack(
        f"<{EMBEDDING_DIMENSIONS}f", *record["embedding"]
    )
    return hash_record


def persist_records(
    records: list[dict[str, Any]],
    client: Redis,
    index_name: str = DEFAULT_INDEX,
    prefix: str = DEFAULT_PREFIX,
    *,
    recreate_index: bool = False,
) -> int:
    if not records:
        raise RedisStoreError("No embedding records were provided")
    seen_ids: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise RedisStoreError(f"Expected a JSON object at record {line_number}")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise RedisStoreError(f"Missing string id at record {line_number}")
        if record_id in seen_ids:
            raise RedisStoreError(f"Duplicate record id {record_id!r}")
        seen_ids.add(record_id)
        _validate_embedding(record, line_number, Path("<records>"))

    try:
        if recreate_index:
            if index_name == DEFAULT_INDEX:
                for legacy_index in LEGACY_INDEXES:
                    _drop_index(client, legacy_index)
            _drop_index(client, index_name)
        _ensure_index(client, index_name, prefix)
        pipeline = client.pipeline(transaction=False)
        pending = 0
        for record in records:
            redis_key = f"{prefix}{record['id']}"
            pipeline.hset(redis_key, mapping=_to_hash(record))
            pending += 1
            if pending == PIPELINE_BATCH_SIZE:
                pipeline.execute()
                pending = 0
        if pending:
            pipeline.execute()
    except RedisError as exc:
        raise RedisStoreError(f"Redis persistence failed: {exc}") from exc

    return len(records)


def persist(
    input_path: Path = DEFAULT_INPUT,
    redis_url: str | None = None,
    index_name: str = DEFAULT_INDEX,
    prefix: str = DEFAULT_PREFIX,
    *,
    recreate_index: bool = False,
) -> int:
    return persist_corpora(
        (input_path,),
        redis_url,
        index_name,
        prefix,
        recreate_index=recreate_index,
    )


def persist_corpora(
    input_paths: tuple[Path, ...],
    redis_url: str | None = None,
    index_name: str = DEFAULT_INDEX,
    prefix: str = DEFAULT_PREFIX,
    *,
    recreate_index: bool = False,
) -> int:
    if not input_paths:
        raise RedisStoreError("At least one embeddings file must be provided")
    if recreate_index and index_name == DEFAULT_INDEX and len(input_paths) < 2:
        raise RedisStoreError(
            "Recreating the shared Getnet index requires all locale corpora. "
            "Use --locale all."
        )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for input_path in input_paths:
        corpus_records = load_embedding_records(input_path)
        duplicates = seen_ids.intersection(record["id"] for record in corpus_records)
        if duplicates:
            raise RedisStoreError(
                f"Duplicate record IDs across embeddings files: {sorted(duplicates)[:5]}"
            )
        seen_ids.update(record["id"] for record in corpus_records)
        records.extend(corpus_records)

    connection_url = _resolve_redis_url(redis_url)
    client = Redis.from_url(connection_url)
    try:
        client.ping()
        count = persist_records(
            records,
            client,
            index_name,
            prefix,
            recreate_index=recreate_index,
        )
    except RedisError as exc:
        raise RedisStoreError(f"Could not connect to or persist data in Redis: {exc}") from exc
    finally:
        client.close()
    LOGGER.info("Persisted %d embedding records to Redis index %s", count, index_name)
    return count


def _resolve_redis_url(redis_url: str | None = None) -> str:
    load_dotenv(ENV_FILE, override=False)
    connection_url = (
        redis_url
        or os.environ.get("GETNET_REDIS_URL")
        or os.environ.get("REDIS_URL")
    )
    if not connection_url:
        raise RedisStoreError(
            f"No Redis URL configured. Set GETNET_REDIS_URL or REDIS_URL in {ENV_FILE}, "
            "or pass --redis-url."
        )
    return connection_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist Getnet locale embeddings to a shared Redis vector index."
    )
    parser.add_argument(
        "--locale",
        choices=ALL_LOCALES,
        default="all",
        help="Embedding corpus to persist (default: all)",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input embeddings JSONL")
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL (defaults to GETNET_REDIS_URL or REDIS_URL from the repository .env)",
    )
    parser.add_argument("--index", default=None, help="Redis Search index name")
    parser.add_argument("--prefix", default=None, help="Redis key prefix")
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Drop the shared index and legacy locale indexes before rebuilding all data",
    )
    args = parser.parse_args()
    locales = SUPPORTED_LOCALES if args.locale == "all" else (args.locale,)
    input_paths = (
        (args.input,)
        if args.input
        else tuple(embeddings_for_locale(locale) for locale in locales)
    )
    index_name = args.index or DEFAULT_INDEX
    prefix = args.prefix or DEFAULT_PREFIX
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        persist_corpora(
            input_paths,
            args.redis_url,
            index_name,
            prefix,
            recreate_index=args.recreate_index,
        )
    except RedisStoreError as exc:
        parser.exit(1, f"Redis persistence failed: {exc}\n")


if __name__ == "__main__":
    main()
