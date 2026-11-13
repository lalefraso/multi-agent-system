import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from retrieval_evaluation.evaluate import (
    EvaluationError,
    HOLDOUT,
    audit_corpora,
    capture_rankings,
    compare,
    evaluate,
    exact_rankings,
    read_json,
    validate_holdout,
)


def fixture():
    return {
        "version": "test-v1",
        "corpora": {"en": "0" * 64},
        "queries": [
            {
                "id": "q1",
                "query": "first query",
                "relevant": [
                    {"id": doc, "locale": "en", "url": "https://example.test", "heading": "Heading", "evidence": "evidence"}
                    for doc in ("a", "b")
                ],
            },
            {
                "id": "q2",
                "query": "second query",
                "relevant": [
                    {"id": "c", "locale": "en", "url": "https://example.test", "heading": "Heading", "evidence": "evidence"}
                ],
            },
        ],
    }


def run(rankings, depth=4):
    return {
        "holdout_version": "test-v1",
        "corpora": {"en": "0" * 64},
        "method": "test",
        "depth": depth,
        "rankings": rankings,
    }


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.holdout = validate_holdout(fixture())

    def test_metrics_and_macro_average_with_multiple_relevant_and_unjudged(self):
        report = evaluate(self.holdout, run({"q1": ["x", "a", "y", "b"], "q2": ["x"]}), 4)
        first = report["queries"]["q1"]
        self.assertEqual(first["recall"], 1)
        self.assertEqual(first["mrr"], 0.5)
        self.assertAlmostEqual(
            first["ndcg"],
            (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3)),
        )
        self.assertEqual(first["unjudged"], 2)
        self.assertEqual(report["queries"]["q2"]["recall"], 0)
        self.assertEqual(report["macro"]["recall"], 0.5)
        self.assertEqual(report["macro"]["mrr"], 0.25)
        self.assertAlmostEqual(report["macro"]["ndcg"], first["ndcg"] / 2)

    def test_cutoff_empty_results_and_partial_recall(self):
        report = evaluate(self.holdout, run({"q1": ["a", "x", "b"], "q2": []}), 2)
        self.assertEqual(report["queries"]["q1"]["recall"], 0.5)
        self.assertEqual(report["queries"]["q1"]["mrr"], 1)
        self.assertAlmostEqual(report["queries"]["q1"]["ndcg"], 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(report["queries"]["q2"]["ndcg"], 0)
        self.assertEqual(report["queries"]["q2"]["unjudged"], 0)

    def test_comparison_uses_same_queries_and_cutoff(self):
        current = run({"q1": ["a", "b"], "q2": ["c"]})
        baseline = run({"q1": ["x"], "q2": []})
        report = compare(self.holdout, current, baseline, 2)
        self.assertEqual(report["macro_delta"], {"recall": 1.0, "mrr": 1.0, "ndcg": 1.0})

    def test_rejects_malformed_rankings_and_cutoffs(self):
        good = run({"q1": ["a"], "q2": []})
        for bad, k in [
            (run({"q1": ["a", "a"], "q2": []}), 2),
            (run({"q1": ["a"]}), 2),
            (run({"q1": ["a"], "q2": [], "extra": []}), 2),
            (run({"q1": ["a"], "q2": []}, depth=1), 2),
            (run({"q1": [None], "q2": []}), 2),
            (run({"q1": ["a"] * 5, "q2": []}), 2),
            ({**good, "holdout_version": "other"}, 2),
            ({**good, "corpora": {"en": "1" * 64}}, 2),
            (good, 0),
            (good, True),
        ]:
            with self.subTest(bad=bad, k=k), self.assertRaises(EvaluationError):
                evaluate(self.holdout, bad, k)

    def test_holdout_rejects_duplicate_and_unbacked_judgments(self):
        bad = fixture()
        bad["queries"][0]["relevant"].append(copy.deepcopy(bad["queries"][0]["relevant"][0]))
        with self.assertRaises(EvaluationError):
            validate_holdout(bad)
        bad = fixture()
        bad["queries"][0]["relevant"][0]["evidence"] = ""
        with self.assertRaises(EvaluationError):
            validate_holdout(bad)
        bad = fixture()
        bad["queries"][1]["id"] = "q1"
        with self.assertRaises(EvaluationError):
            validate_holdout(bad)

    def test_corpus_audit_checks_fingerprint_and_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            records = [
                {"id": doc, "url": "https://example.test", "section_heading": "Heading", "text": "source evidence"}
                for doc in ("a", "b", "c")
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            self.holdout["corpora"]["en"] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(audit_corpora(self.holdout, {"en": path})), 3)
            self.holdout["queries"][0]["relevant"][0]["evidence"] = "not in source"
            with self.assertRaisesRegex(EvaluationError, "Evidence mismatch"):
                audit_corpora(self.holdout, {"en": path})
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "fingerprint mismatch"):
                audit_corpora(self.holdout, {"en": path})

    def test_capture_preserves_ranking_and_rejects_missing_ids(self):
        calls = []

        def search(query, depth):
            calls.append((query, depth))
            return [{"id": "b"}, {"id": "a"}] if query == "first query" else []

        result = capture_rankings(self.holdout, 2, search)
        self.assertEqual(result["rankings"], {"q1": ["b", "a"], "q2": []})
        self.assertEqual(calls, [("first query", 2), ("second query", 2)])
        with self.assertRaisesRegex(EvaluationError, "without an ID"):
            capture_rankings(self.holdout, 2, lambda _q, _k: [{}])

    def test_exact_cosine_baseline_uses_query_prefix_and_id_tie_break(self):
        def record(vector):
            return {
                "embedding_model": "intfloat/multilingual-e5-small",
                "embedding_dimensions": 384,
                "embedding": vector + [0.0] * 382,
            }

        records = {"b": record([1.0, 0.0]), "a": record([1.0, 0.0]), "c": record([0.0, 1.0])}
        calls = []

        def encode(text):
            calls.append(text)
            return [1.0, 0.0] + [0.0] * 382

        result = exact_rankings(self.holdout, records, 2, encode)
        self.assertEqual(result["rankings"]["q1"], ["a", "b"])
        self.assertEqual(calls, ["query: first query", "query: second query"])
        records["c"]["embedding"][0] = float("nan")
        with self.assertRaisesRegex(EvaluationError, "non-finite"):
            exact_rankings(self.holdout, records, 2, encode)

    def test_versioned_holdout_is_valid(self):
        holdout = validate_holdout(read_json(HOLDOUT))
        self.assertEqual(holdout["version"], "getnet-retrieval-holdout-v1")
        baseline = read_json(HOLDOUT.with_name("exact-cosine.v1.json"))
        report = evaluate(holdout, baseline, 5)
        self.assertEqual(len(report["queries"]), 6)
        self.assertAlmostEqual(report["macro"]["recall"], 16 / 27)
        self.assertAlmostEqual(report["macro"]["mrr"], 2 / 3)
        self.assertAlmostEqual(report["macro"]["ndcg"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
