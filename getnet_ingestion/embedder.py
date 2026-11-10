from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any


MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384
DEFAULT_INPUT = Path(__file__).parent / "data" / "getnet_en_sections.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "getnet_en_embeddings.jsonl"
BATCH_SIZE = 32
PASSAGE_PREFIX = "passage: "

LOGGER = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when source records cannot be embedded without data loss."""


def load_records(input_path: Path) -> list[dict[str, Any]]:
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
                    raise EmbeddingError(
                        f"Invalid JSON on line {line_number} of {input_path}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise EmbeddingError(
                        f"Expected a JSON object on line {line_number} of {input_path}"
                    )
                record_id = record.get("id")
                text = record.get("text")
                if not isinstance(record_id, str) or not record_id:
                    raise EmbeddingError(f"Missing string id on line {line_number} of {input_path}")
                if not isinstance(text, str) or not text.strip():
                    raise EmbeddingError(
                        f"Missing non-empty text for record {record_id!r} on line {line_number}"
                    )
                if record_id in seen_ids:
                    raise EmbeddingError(f"Duplicate record id {record_id!r} in {input_path}")
                seen_ids.add(record_id)
                records.append(record)
    except OSError as exc:
        raise EmbeddingError(f"Could not read corpus {input_path}: {exc}") from exc

    if not records:
        raise EmbeddingError(f"Corpus {input_path} contains no records")
    return records


def _token_count(model: Any, text: str) -> int:
    tokenized = model.tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    input_ids = tokenized["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise EmbeddingError("Tokenizer returned an unexpected batch for one section")
        input_ids = input_ids[0]
    return len(input_ids)


def _passage_text(text: str) -> str:
    return f"{PASSAGE_PREFIX}{text}"


def _fit_semantic_blocks(
    heading: str,
    blocks: list[str],
    model: Any,
    max_tokens: int,
    record_label: str,
) -> list[str]:
    fitted: list[str] = []
    for block in blocks:
        block_text = f"{heading}\n{block}"
        if _token_count(model, _passage_text(block_text)) <= max_tokens:
            fitted.append(block_text)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", block.strip())
        if len(sentences) < 2:
            token_count = _token_count(model, _passage_text(block_text))
            raise EmbeddingError(
                f"Semantic block in {record_label!r} exceeds the model limit of "
                f"{max_tokens} tokens ({token_count}). It was not divided or truncated."
            )

        sentence_group: list[str] = []
        for sentence in sentences:
            candidate = " ".join((*sentence_group, sentence))
            candidate_text = f"{heading}\n{candidate}"
            if _token_count(model, _passage_text(candidate_text)) <= max_tokens:
                sentence_group.append(sentence)
                continue
            if not sentence_group:
                token_count = _token_count(model, _passage_text(candidate_text))
                raise EmbeddingError(
                    f"Sentence in {record_label!r} exceeds the model limit of "
                    f"{max_tokens} tokens ({token_count}). It was not divided or truncated."
                )
            fitted.append(f"{heading}\n{' '.join(sentence_group)}")
            sentence_group = [sentence]
            single_sentence_text = f"{heading}\n{sentence}"
            token_count = _token_count(model, _passage_text(single_sentence_text))
            if token_count > max_tokens:
                raise EmbeddingError(
                    f"Sentence in {record_label!r} exceeds the model limit of "
                    f"{max_tokens} tokens ({token_count}). It was not divided or truncated."
                )
        if sentence_group:
            fitted.append(f"{heading}\n{' '.join(sentence_group)}")
    return fitted


def _separate_semantic_blocks(
    records: list[dict[str, Any]],
    model: Any,
    max_tokens: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    split_sections = 0

    for record in records:
        text = record["text"]
        if _token_count(model, _passage_text(text)) <= max_tokens:
            result.append(record)
            continue

        lines = text.splitlines()
        if len(lines) < 2:
            result.append(record)
            continue

        heading = lines[0]
        blocks = [line.strip() for line in lines[1:] if line.strip()]
        fitted_blocks = _fit_semantic_blocks(
            heading,
            blocks,
            model,
            max_tokens,
            record.get("section_heading", record["id"]),
        )
        block_records = []
        for block_index, block_text in enumerate(fitted_blocks):
            block_id = hashlib.sha256(
                f"{record['id']}\n{block_index}\n{block_text}".encode("utf-8")
            ).hexdigest()
            block_records.append({**record, "id": block_id, "text": block_text})

        result.extend(block_records)
        split_sections += 1

    LOGGER.info("Separated %d long sections at semantic block boundaries", split_sections)
    return result


def vectorize_records(
    records: list[dict[str, Any]],
    model: Any,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    dimensions = model.get_embedding_dimension()
    if dimensions != EMBEDDING_DIMENSIONS:
        raise EmbeddingError(
            f"Expected {EMBEDDING_DIMENSIONS}-dimensional embeddings, got {dimensions}"
        )

    max_tokens = model.max_seq_length
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise EmbeddingError(f"Invalid model max sequence length: {max_tokens!r}")

    records_to_embed = _separate_semantic_blocks(records, model, max_tokens)
    oversized = []
    for record in records_to_embed:
        token_count = _token_count(model, _passage_text(record["text"]))
        if token_count > max_tokens:
            oversized.append((record, token_count))
    if oversized:
        details = "; ".join(
            f"{record.get('section_heading', 'Unlabeled section')!r} "
            f"[{record['id']}]: {token_count} tokens"
            for record, token_count in oversized[:20]
        )
        remaining = len(oversized) - 20
        if remaining > 0:
            details += f"; and {remaining} more"
        raise EmbeddingError(
            f"{len(oversized)} semantic block(s) exceed the model limit of {max_tokens} "
            f"tokens. No text was truncated: {details}"
        )

    vectors = model.encode(
        [_passage_text(record["text"]) for record in records_to_embed],
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if len(vectors) != len(records_to_embed):
        raise EmbeddingError(
            f"Model returned {len(vectors)} vectors for {len(records_to_embed)} records"
        )

    embedded_records = []
    for record, vector in zip(records_to_embed, vectors):
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingError(
                f"Record {record['id']!r} has a {len(vector)}-dimensional vector; "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
        embedding = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in embedding):
            raise EmbeddingError(f"Model returned non-finite vector values for {record['id']!r}")
        embedded_records.append(
            {
                **record,
                "embedding_model": model_name,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "embedding": embedding,
            }
        )
    return embedded_records


def write_records(records: list[dict[str, Any]], output_path: Path) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary_path.replace(output_path)
    except (OSError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise EmbeddingError(f"Could not write embeddings to {output_path}: {exc}") from exc


def vectorize(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    model_name: str = MODEL_NAME,
) -> int:
    records = load_records(input_path)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "sentence-transformers is not installed; install getnet_ingestion/requirements.txt"
        ) from exc

    LOGGER.info("Loading embedding model %s", model_name)
    model = SentenceTransformer(model_name)
    embedded_records = vectorize_records(records, model, model_name)
    write_records(embedded_records, output_path)
    LOGGER.info("Saved %d embeddings to %s", len(embedded_records), output_path)
    return len(embedded_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create whole-section embeddings for a Getnet locale corpus."
    )
    parser.add_argument(
        "--locale",
        choices=("en", "mx"),
        default="en",
        help="Corpus locale (default: en)",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input corpus JSONL")
    parser.add_argument("--output", type=Path, default=None, help="Output embeddings JSONL")
    parser.add_argument("--model", default=MODEL_NAME, help="Sentence Transformers model name")
    args = parser.parse_args()
    input_path = args.input or Path(__file__).parent / "data" / f"getnet_{args.locale}_sections.jsonl"
    output_path = args.output or Path(__file__).parent / "data" / f"getnet_{args.locale}_embeddings.jsonl"
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        vectorize(input_path, output_path, args.model)
    except EmbeddingError as exc:
        parser.exit(1, f"Vectorization failed: {exc}\n")


if __name__ == "__main__":
    main()
