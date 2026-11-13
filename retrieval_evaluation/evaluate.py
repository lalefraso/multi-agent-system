from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Callable


HOLDOUT = Path(__file__).with_name("holdout.v1.json")
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSIONS = 384


class EvaluationError(ValueError):
    """Raised when evaluation inputs are incomplete or inconsistent."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Cannot read JSON from {path}: {exc}") from exc


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_holdout(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not _nonempty(data.get("version")):
        raise EvaluationError("Holdout must have a non-empty version")
    corpora = data.get("corpora")
    if not isinstance(corpora, dict) or not corpora or any(
        not _nonempty(locale)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        for locale, digest in corpora.items()
    ):
        raise EvaluationError("Holdout must have corpus SHA-256 fingerprints")
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise EvaluationError("Holdout must contain queries")
    seen_queries: set[str] = set()
    for query in queries:
        if not isinstance(query, dict) or not _nonempty(query.get("id")) or not _nonempty(query.get("query")):
            raise EvaluationError("Each query must have a non-empty id and query")
        query_id = query["id"]
        if query_id in seen_queries:
            raise EvaluationError(f"Duplicate query id: {query_id}")
        seen_queries.add(query_id)
        relevant = query.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise EvaluationError(f"Query {query_id} must have relevant documents")
        seen_docs: set[str] = set()
        for judgment in relevant:
            if not isinstance(judgment, dict) or any(
                not _nonempty(judgment.get(field))
                for field in ("id", "locale", "url", "heading", "evidence")
            ):
                raise EvaluationError(f"Invalid judgment for {query_id}")
            if judgment["locale"] not in corpora:
                raise EvaluationError(f"Unknown corpus locale for {query_id}")
            if judgment["id"] in seen_docs:
                raise EvaluationError(f"Duplicate judgment for {query_id}: {judgment['id']}")
            seen_docs.add(judgment["id"])
    return data


def audit_corpora(holdout: dict[str, Any], paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    if set(paths) != set(holdout["corpora"]):
        raise EvaluationError("Provide exactly the corpus locales listed in the holdout")
    records: dict[str, dict[str, Any]] = {}
    for locale, path in paths.items():
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise EvaluationError(f"Cannot read corpus {path}: {exc}") from exc
        if hashlib.sha256(raw).hexdigest() != holdout["corpora"][locale]:
            raise EvaluationError(f"Corpus fingerprint mismatch for {locale}: {path}")
        try:
            lines = raw.decode("utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not _nonempty(record.get("id")):
                    raise EvaluationError(f"Invalid record in {path}")
                if record["id"] in records:
                    raise EvaluationError(f"Duplicate corpus id: {record['id']}")
                records[record["id"]] = {**record, "_locale": locale}
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"Invalid corpus {path}: {exc}") from exc
    for query in holdout["queries"]:
        for judgment in query["relevant"]:
            record = records.get(judgment["id"])
            if (
                record is None
                or record["_locale"] != judgment["locale"]
                or record.get("url") != judgment["url"]
                or record.get("section_heading") != judgment["heading"]
                or not isinstance(record.get("text"), str)
                or judgment["evidence"] not in record["text"]
            ):
                raise EvaluationError(f"Evidence mismatch for {query['id']}: {judgment['id']}")
    return records


def validate_run(holdout: dict[str, Any], run: Any, k: int) -> dict[str, list[str]]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise EvaluationError("k must be a positive integer")
    if not isinstance(run, dict) or run.get("holdout_version") != holdout["version"]:
        raise EvaluationError("Ranking holdout version does not match")
    if run.get("corpora") != holdout["corpora"] or not _nonempty(run.get("method")):
        raise EvaluationError("Ranking must identify its method and the holdout corpus fingerprints")
    depth = run.get("depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < k:
        raise EvaluationError("Ranking depth must be an integer at least k")
    rankings = run.get("rankings")
    expected = {query["id"] for query in holdout["queries"]}
    if not isinstance(rankings, dict) or set(rankings) != expected:
        raise EvaluationError("Rankings must contain exactly the holdout query IDs")
    for query_id, ids in rankings.items():
        if (
            not isinstance(ids, list)
            or len(ids) > depth
            or any(not _nonempty(item) for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise EvaluationError(f"Invalid or duplicate ranked IDs for {query_id}")
    return rankings


def evaluate(holdout: dict[str, Any], run: Any, k: int) -> dict[str, Any]:
    rankings = validate_run(holdout, run, k)
    per_query: dict[str, dict[str, float | int]] = {}
    for query in holdout["queries"]:
        relevant = {item["id"] for item in query["relevant"]}
        ranked = rankings[query["id"]][:k]
        hits = [index for index, doc_id in enumerate(ranked, start=1) if doc_id in relevant]
        dcg = sum(1 / math.log2(rank + 1) for rank in hits)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        per_query[query["id"]] = {
            "recall": len(hits) / len(relevant),
            "mrr": 1 / hits[0] if hits else 0.0,
            "ndcg": dcg / ideal,
            "judged_relevant": len(hits),
            "unjudged": len(ranked) - len(hits),
        }
    averages = {
        metric: sum(float(result[metric]) for result in per_query.values()) / len(per_query)
        for metric in ("recall", "mrr", "ndcg")
    }
    return {
        "holdout_version": holdout["version"],
        "corpora": holdout["corpora"],
        "method": run["method"],
        "depth": run["depth"],
        "k": k,
        "macro": averages,
        "queries": per_query,
    }


def compare(holdout: dict[str, Any], current: Any, baseline: Any, k: int) -> dict[str, Any]:
    current_report = evaluate(holdout, current, k)
    baseline_report = evaluate(holdout, baseline, k)
    return {
        "current": current_report,
        "baseline": baseline_report,
        "macro_delta": {
            metric: current_report["macro"][metric] - baseline_report["macro"][metric]
            for metric in ("recall", "mrr", "ndcg")
        },
    }


def capture_rankings(
    holdout: dict[str, Any], depth: int, search: Callable[[str, int], list[dict[str, Any]]]
) -> dict[str, Any]:
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 10:
        raise EvaluationError("Current retrieval depth must be between 1 and 10")
    rankings: dict[str, list[str]] = {}
    for query in holdout["queries"]:
        results = search(query["query"], depth)
        if not isinstance(results, list) or any(not isinstance(item, dict) or not _nonempty(item.get("id")) for item in results):
            raise EvaluationError(f"Retrieval returned a result without an ID for {query['id']}")
        rankings[query["id"]] = [item["id"] for item in results]
    run = {
        "holdout_version": holdout["version"],
        "corpora": holdout["corpora"],
        "method": "current-vector-search",
        "depth": depth,
        "rankings": rankings,
    }
    validate_run(holdout, run, depth)
    return run


def exact_rankings(
    holdout: dict[str, Any],
    records: dict[str, dict[str, Any]],
    depth: int,
    encode: Callable[[str], Any],
) -> dict[str, Any]:
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise EvaluationError("Exact baseline depth must be positive")

    def vector(values: Any) -> tuple[float, ...]:
        if len(values) != DIMENSIONS:
            raise EvaluationError(f"Expected {DIMENSIONS}-dimensional vectors")
        try:
            result = struct.unpack(f"<{DIMENSIONS}f", struct.pack(f"<{DIMENSIONS}f", *values))
        except (OverflowError, TypeError, struct.error) as exc:
            raise EvaluationError(f"Invalid vector: {exc}") from exc
        if not all(math.isfinite(value) for value in result):
            raise EvaluationError("Vector contains non-finite values")
        return result

    vectors = []
    for record_id, record in records.items():
        if record.get("embedding_model") != MODEL_NAME or record.get("embedding_dimensions") != DIMENSIONS:
            raise EvaluationError(f"Unexpected embedding model or dimensions for {record_id}")
        values = record.get("embedding")
        if not isinstance(values, list):
            raise EvaluationError(f"Missing embedding for {record_id}")
        embedding = vector(values)
        norm = math.sqrt(sum(value * value for value in embedding))
        if not norm:
            raise EvaluationError(f"Zero embedding for {record_id}")
        vectors.append((record_id, embedding, norm))

    rankings = {}
    for query in holdout["queries"]:
        query_vector = vector(encode(f"query: {query['query']}"))
        norm = math.sqrt(sum(value * value for value in query_vector))
        if not norm:
            raise EvaluationError(f"Zero query embedding for {query['id']}")
        scores = [
            (sum(a * b for a, b in zip(query_vector, embedding, strict=True)) / (norm * doc_norm), record_id)
            for record_id, embedding, doc_norm in vectors
        ]
        scores.sort(key=lambda item: (-item[0], item[1]))
        rankings[query["id"]] = [record_id for _, record_id in scores[:depth]]
    run = {
        "holdout_version": holdout["version"],
        "corpora": holdout["corpora"],
        "method": f"exact-cosine-{MODEL_NAME}@{MODEL_REVISION}",
        "depth": depth,
        "rankings": rankings,
    }
    validate_run(holdout, run, depth)
    return run


def verify_index(records: dict[str, dict[str, Any]]) -> None:
    from dotenv import load_dotenv
    from redis import Redis
    from redis.commands.search.query import Query

    from app.agents.knowledge import ENV_FILE, _knowledge_settings
    from getnet_ingestion.redis_store import DEFAULT_INDEX, DEFAULT_PREFIX, _resolve_redis_url

    load_dotenv(ENV_FILE, override=False)
    client = Redis.from_url(_resolve_redis_url())
    try:
        index = client.ft(_knowledge_settings().get("vector_index", DEFAULT_INDEX))
        ids: set[str] = set()
        offset = 0
        while True:
            response = index.search(Query("*").no_content().paging(offset, 500).dialect(2))
            for doc in response.docs:
                key = doc.id.decode("utf-8") if isinstance(doc.id, bytes) else doc.id
                if not isinstance(key, str) or not key.startswith(DEFAULT_PREFIX):
                    raise EvaluationError(f"Unexpected indexed document key: {key!r}")
                ids.add(key.removeprefix(DEFAULT_PREFIX))
            offset += len(response.docs)
            if offset >= response.total:
                break
            if not response.docs:
                raise EvaluationError("Index pagination ended before all documents were read")
        if ids != set(records) or offset != len(records):
            raise EvaluationError("Redis index documents do not match the holdout corpus snapshots")
        items = list(records.items())
        for start in range(0, len(items), 500):
            batch = items[start:start + 500]
            pipeline = client.pipeline(transaction=False)
            for record_id, _ in batch:
                pipeline.hgetall(DEFAULT_PREFIX + record_id)
            for (record_id, record), fields in zip(batch, pipeline.execute(), strict=True):
                for field in ("url", "page_title", "section_heading", "category", "language", "text"):
                    value = record.get(field)
                    if isinstance(value, str) and fields.get(field.encode()) != value.encode():
                        raise EvaluationError(f"Redis {field} mismatch for {record_id}")
                vector = record.get("embedding")
                if not isinstance(vector, list) or len(vector) != DIMENSIONS:
                    raise EvaluationError(f"Invalid embedding in corpus for {record_id}")
                if fields.get(b"embedding") != struct.pack(f"<{DIMENSIONS}f", *vector):
                    raise EvaluationError(f"Redis embedding mismatch for {record_id}")
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ranked RAG retrieval, without answer generation")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "capture-current", "capture-exact"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--en", type=Path, required=True, help="English embeddings JSONL snapshot")
        sub.add_argument("--mx", type=Path, required=True, help="Mexico embeddings JSONL snapshot")
        if name != "audit":
            sub.add_argument("--depth", type=int, default=5)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--rankings", type=Path, required=True)
    scoring.add_argument("--baseline", type=Path)
    scoring.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    try:
        holdout = validate_holdout(read_json(args.holdout))
        if args.command == "score":
            run = read_json(args.rankings)
            output = (
                compare(holdout, run, read_json(args.baseline), args.k)
                if args.baseline else evaluate(holdout, run, args.k)
            )
        else:
            records = audit_corpora(holdout, {"en": args.en, "mx": args.mx})
            if args.command == "audit":
                output = {"holdout_version": holdout["version"], "documents": len(records), "status": "verified"}
            elif args.command == "capture-current":
                verify_index(records)
                from app.agents.knowledge import vector_search

                def search(query: str, depth: int) -> list[dict[str, Any]]:
                    return json.loads(vector_search.invoke({"query": query, "result_count": depth}))

                output = capture_rankings(holdout, args.depth, search)
            else:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(
                    MODEL_NAME, revision=MODEL_REVISION, local_files_only=True, device="cpu"
                )
                output = exact_rankings(
                    holdout, records, args.depth,
                    lambda text: model.encode(text, convert_to_numpy=True, normalize_embeddings=True),
                )
        print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    except EvaluationError as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
