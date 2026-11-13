"""Score recorded runs and human judgments without invoking a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROUTES = {"knowledge", "customer_support", "human_escalation"}
DIMENSIONS = ("groundedness", "factual_correctness", "relevance", "usefulness")
VERDICTS = {"pass", "fail", "not_applicable"}


class EvaluationError(ValueError):
    pass


def _object(value, required, location):
    if not isinstance(value, dict) or set(value) != set(required):
        raise EvaluationError(f"{location}: expected fields {', '.join(required)}")
    return value


def _text(value, location):
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{location}: expected non-empty text")
    return value


def _list(value, location):
    if not isinstance(value, list):
        raise EvaluationError(f"{location}: expected a list")
    return value


def _routes(value, location):
    routes = _list(value, location)
    if any(not isinstance(route, str) or route not in ROUTES for route in routes):
        raise EvaluationError(f"{location}: unknown route")
    if len(set(routes)) != len(routes):
        raise EvaluationError(f"{location}: duplicate route")
    return set(routes)


def validate_dataset(dataset):
    _object(dataset, ("schema_version", "dataset_id", "label_status", "label_review", "cases"), "dataset")
    if type(dataset["schema_version"]) is not int or dataset["schema_version"] != 1:
        raise EvaluationError("dataset: unsupported schema_version")
    _text(dataset["dataset_id"], "dataset_id")
    if dataset["label_status"] not in ("proposed", "human_reviewed"):
        raise EvaluationError("dataset: invalid label_status")
    if dataset["label_status"] == "human_reviewed":
        _text(dataset["label_review"], "dataset.label_review")
    elif dataset["label_review"] is not None:
        raise EvaluationError("dataset: proposed labels must not have label_review")
    cases = {}
    for case in _list(dataset["cases"], "cases"):
        _object(case, ("id", "message", "expected_routes", "rationale"), "case")
        case_id = _text(case["id"], "case.id")
        if case_id in cases:
            raise EvaluationError(f"duplicate case id: {case_id}")
        _text(case["message"], f"{case_id}.message")
        _text(case["rationale"], f"{case_id}.rationale")
        if case["expected_routes"] is not None:
            _routes(case["expected_routes"], f"{case_id}.expected_routes")
        cases[case_id] = case
    if not cases:
        raise EvaluationError("dataset: cases must not be empty")
    return cases


def _validate_review(review, evidence, location):
    _object(review, ("reviewer", "dimensions"), location)
    reviewer = _text(review["reviewer"], f"{location}.reviewer")
    dimensions = review["dimensions"]
    _object(dimensions, DIMENSIONS, f"{location}.dimensions")
    for name in DIMENSIONS:
        judgment = dimensions[name]
        _object(judgment, ("verdict", "rationale", "citations"), f"{location}.{name}")
        verdict = judgment["verdict"]
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise EvaluationError(f"{location}.{name}: invalid verdict")
        _text(judgment["rationale"], f"{location}.{name}.rationale")
        citations = _list(judgment["citations"], f"{location}.{name}.citations")
        for citation in citations:
            _object(citation, ("evidence_id", "quote"), f"{location}.{name}.citation")
            source_id = _text(citation["evidence_id"], "citation.evidence_id")
            quote = _text(citation["quote"], "citation.quote")
            if source_id not in evidence or quote not in evidence[source_id]["text"]:
                raise EvaluationError(f"{location}.{name}: citation not found in evidence")
        required_kind = {"groundedness": "agent_context", "factual_correctness": "reference"}.get(name)
        if verdict == "pass" and required_kind and not any(
            evidence[citation["evidence_id"]]["kind"] == required_kind for citation in citations
        ):
            raise EvaluationError(f"{location}.{name}: pass requires {required_kind} evidence citation")
    return reviewer


def evaluate(dataset, run, dataset_sha256, run_sha256):
    cases = validate_dataset(dataset)
    _object(run, ("schema_version", "dataset_id", "dataset_sha256", "run_id", "system", "samples"), "run")
    if type(run["schema_version"]) is not int or run["schema_version"] != 1:
        raise EvaluationError("run: unsupported schema_version")
    if run["dataset_id"] != dataset["dataset_id"] or run["dataset_sha256"] != dataset_sha256:
        raise EvaluationError("run: dataset identity or SHA-256 does not match")
    _text(run["run_id"], "run_id")
    _object(run["system"], ("revision", "model", "configuration"), "system")
    for key in ("revision", "model", "configuration"):
        _text(run["system"][key], f"system.{key}")

    sample_keys = set()
    observed_cases = set()
    route_matches = []
    counts = {name: {"pass": 0, "fail": 0, "not_applicable": 0} for name in DIMENSIONS}
    pairs = {name: {"agree": 0, "compared": 0} for name in DIMENSIONS}
    details = []
    samples = _list(run["samples"], "samples")
    if not samples:
        raise EvaluationError("run: samples must not be empty")
    for sample in samples:
        _object(sample, ("case_id", "sample_id", "answer", "routes", "evidence", "reviews"), "sample")
        case_id = _text(sample["case_id"], "sample.case_id")
        sample_id = _text(sample["sample_id"], "sample.sample_id")
        if case_id not in cases:
            raise EvaluationError(f"unknown case id: {case_id}")
        if (case_id, sample_id) in sample_keys:
            raise EvaluationError(f"duplicate sample: {case_id}/{sample_id}")
        sample_keys.add((case_id, sample_id))
        observed_cases.add(case_id)
        _text(sample["answer"], f"{case_id}/{sample_id}.answer")
        routes = _routes(sample["routes"], f"{case_id}/{sample_id}.routes")
        evidence = {}
        for source in _list(sample["evidence"], "sample.evidence"):
            _object(source, ("id", "kind", "origin", "text"), "evidence")
            source_id = _text(source["id"], "evidence.id")
            if source_id in evidence:
                raise EvaluationError(f"duplicate evidence id: {source_id}")
            if source["kind"] not in ("agent_context", "reference"):
                raise EvaluationError("evidence: invalid kind")
            _text(source["origin"], "evidence.origin")
            _text(source["text"], "evidence.text")
            evidence[source_id] = source
        reviews = _list(sample["reviews"], "sample.reviews")
        if len(reviews) > 2:
            raise EvaluationError("sample: at most two independent reviews are supported")
        reviewers = set()
        for review in reviews:
            reviewer = _validate_review(review, evidence, f"{case_id}/{sample_id}.review")
            if reviewer in reviewers:
                raise EvaluationError("sample: duplicate reviewer")
            reviewers.add(reviewer)

        expected = cases[case_id]["expected_routes"]
        match = None if expected is None else routes == set(expected)
        if match is not None:
            route_matches.append(match)
        # Primary review supplies the metric; the second is only for calibration.
        if reviews:
            for name in DIMENSIONS:
                counts[name][reviews[0]["dimensions"][name]["verdict"]] += 1
                if len(reviews) == 2:
                    first = reviews[0]["dimensions"][name]["verdict"]
                    second = reviews[1]["dimensions"][name]["verdict"]
                    if first != "not_applicable" and second != "not_applicable":
                        pairs[name]["compared"] += 1
                        pairs[name]["agree"] += first == second
        details.append({
            "case_id": case_id,
            "sample_id": sample_id,
            "expected_routes": expected,
            "observed_routes": sorted(routes),
            "routing_match": match,
            "review_count": len(reviews),
            "primary_verdicts": {
                name: reviews[0]["dimensions"][name]["verdict"] for name in DIMENSIONS
            } if reviews else None,
        })

    metrics = {}
    for name in DIMENSIONS:
        count = counts[name]
        assessed = count["pass"] + count["fail"]
        metrics[name] = {**count, "assessed": assessed, "pass_rate": count["pass"] / assessed if assessed else None}
    return {
        "schema_version": 1,
        "dataset_id": dataset["dataset_id"],
        "label_status": dataset["label_status"],
        "label_review": dataset["label_review"],
        "dataset_sha256": dataset_sha256,
        "run_sha256": run_sha256,
        "run_id": run["run_id"],
        "system": run["system"],
        "coverage": {"dataset_cases": len(cases), "observed_cases": len(observed_cases), "missing_case_ids": sorted(set(cases) - observed_cases), "samples": len(samples), "reviewed_samples": sum(item["review_count"] > 0 for item in details)},
        "routing": {"matched": sum(route_matches), "assessed": len(route_matches), "unlabeled_samples": len(samples) - len(route_matches), "accuracy": sum(route_matches) / len(route_matches) if route_matches else None},
        "quality": metrics,
        "calibration": {name: {**pair, "agreement": pair["agree"] / pair["compared"] if pair["compared"] else None} for name, pair in pairs.items()},
        "samples": details,
    }


def _load(path):
    try:
        data = Path(path).read_bytes()
        return json.loads(data), hashlib.sha256(data).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: {exc}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    try:
        dataset, dataset_hash = _load(args.dataset)
        run, run_hash = _load(args.run)
        report = evaluate(dataset, run, dataset_hash, run_hash)
    except EvaluationError as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
