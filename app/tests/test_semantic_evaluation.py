import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.semantic import DIMENSIONS, EvaluationError, evaluate, validate_dataset


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation" / "cases" / "v1.json"


def fixture():
    dataset_bytes = DATASET.read_bytes()
    dataset = json.loads(dataset_bytes)
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    judgment = {
        "verdict": "pass",
        "rationale": "Synthetic test judgment.",
        "citations": [{"evidence_id": "e1", "quote": "test evidence"}],
    }
    review = {
        "reviewer": "test-reviewer-1",
        "dimensions": {name: copy.deepcopy(judgment) for name in DIMENSIONS},
    }
    review["dimensions"]["factual_correctness"]["citations"][0]["evidence_id"] = "e2"
    sample = {
        "case_id": "product-comparison",
        "sample_id": "1",
        "answer": "Synthetic test answer.",
        "routes": ["knowledge"],
        "evidence": [
            {"id": "e1", "kind": "agent_context", "origin": "synthetic", "text": "test evidence"},
            {"id": "e2", "kind": "reference", "origin": "synthetic", "text": "test evidence"},
        ],
        "reviews": [review],
    }
    run = {
        "schema_version": 1,
        "dataset_id": dataset["dataset_id"],
        "dataset_sha256": digest,
        "run_id": "synthetic-test",
        "system": {"revision": "test", "model": "none", "configuration": "test"},
        "samples": [sample],
    }
    return dataset, run, digest


class SemanticEvaluationTests(unittest.TestCase):
    def test_scenarios_are_proposed_and_web_routes_are_unlabeled(self):
        dataset, _, _ = fixture()
        cases = validate_dataset(dataset)
        self.assertEqual(len(cases), 10)
        self.assertEqual(dataset["label_status"], "proposed")
        self.assertIsNone(cases["weather"]["expected_routes"])
        self.assertIsNone(cases["exchange-rate"]["expected_routes"])

    def test_metrics_and_missing_coverage_are_explicit(self):
        dataset, run, digest = fixture()
        other = copy.deepcopy(run["samples"][0])
        other.update(case_id="weather", sample_id="1", routes=[], reviews=[])
        run["samples"].append(other)
        repeat = copy.deepcopy(run["samples"][0])
        repeat.update(sample_id="2", routes=["customer_support"], reviews=[])
        run["samples"].append(repeat)
        report = evaluate(dataset, run, digest, "run-hash")
        self.assertEqual(report["coverage"]["dataset_cases"], 10)
        self.assertEqual(report["coverage"]["observed_cases"], 2)
        self.assertEqual(len(report["coverage"]["missing_case_ids"]), 8)
        self.assertEqual(report["coverage"]["samples"], 3)
        self.assertEqual(report["coverage"]["reviewed_samples"], 1)
        self.assertEqual(report["routing"], {
            "matched": 1, "assessed": 2, "unlabeled_samples": 1, "accuracy": 0.5,
        })
        self.assertEqual(report["quality"]["groundedness"]["pass_rate"], 1.0)
        self.assertEqual(report["quality"]["groundedness"]["assessed"], 1)
        self.assertEqual(report["label_status"], "proposed")
        self.assertIsNone(report["samples"][1]["routing_match"])

    def test_calibration_compares_independent_reviews_without_double_counting(self):
        dataset, run, digest = fixture()
        second = copy.deepcopy(run["samples"][0]["reviews"][0])
        second["reviewer"] = "test-reviewer-2"
        second["dimensions"]["groundedness"]["verdict"] = "fail"
        second["dimensions"]["usefulness"]["verdict"] = "not_applicable"
        run["samples"][0]["reviews"].append(second)
        report = evaluate(dataset, run, digest, "hash")
        self.assertEqual(report["calibration"]["groundedness"], {"agree": 0, "compared": 1, "agreement": 0.0})
        self.assertEqual(report["calibration"]["relevance"]["agreement"], 1.0)
        self.assertIsNone(report["calibration"]["usefulness"]["agreement"])
        self.assertEqual(report["quality"]["groundedness"]["assessed"], 1)

    def test_no_reviews_or_labels_yield_null_not_perfect_score(self):
        dataset, run, digest = fixture()
        run["samples"][0].update(case_id="weather", reviews=[])
        report = evaluate(dataset, run, digest, "hash")
        self.assertIsNone(report["routing"]["accuracy"])
        self.assertIsNone(report["quality"]["relevance"]["pass_rate"])
        self.assertEqual(report["quality"]["relevance"]["assessed"], 0)

    def test_rejects_bad_evidence_and_unsubstantiated_pass(self):
        dataset, run, digest = fixture()
        judgment = run["samples"][0]["reviews"][0]["dimensions"]["groundedness"]
        judgment["citations"][0]["quote"] = "not in source"
        with self.assertRaisesRegex(EvaluationError, "citation not found"):
            evaluate(dataset, run, digest, "hash")
        judgment["citations"] = []
        with self.assertRaisesRegex(EvaluationError, "pass requires agent_context evidence"):
            evaluate(dataset, run, digest, "hash")
        judgment["verdict"] = "fail"
        evaluate(dataset, run, digest, "hash")

    def test_correctness_pass_requires_reference_not_just_agent_context(self):
        dataset, run, digest = fixture()
        judgment = run["samples"][0]["reviews"][0]["dimensions"]["factual_correctness"]
        judgment["citations"][0]["evidence_id"] = "e1"
        with self.assertRaisesRegex(EvaluationError, "pass requires reference evidence"):
            evaluate(dataset, run, digest, "hash")

    def test_routing_is_exact_set_match_and_not_order_sensitive(self):
        dataset, run, digest = fixture()
        dataset["cases"][0]["expected_routes"] = ["knowledge", "customer_support"]
        run["samples"][0]["routes"] = ["customer_support", "knowledge"]
        self.assertTrue(evaluate(dataset, run, digest, "hash")["samples"][0]["routing_match"])
        run["samples"][0]["routes"].append("human_escalation")
        self.assertFalse(evaluate(dataset, run, digest, "hash")["samples"][0]["routing_match"])

    def test_rejects_mismatched_dataset_and_invalid_records(self):
        dataset, run, digest = fixture()
        run["dataset_sha256"] = "wrong"
        with self.assertRaisesRegex(EvaluationError, "does not match"):
            evaluate(dataset, run, digest, "hash")
        run["dataset_sha256"] = digest
        run["samples"].append(copy.deepcopy(run["samples"][0]))
        with self.assertRaisesRegex(EvaluationError, "duplicate sample"):
            evaluate(dataset, run, digest, "hash")
        run["samples"].pop()
        run["samples"][0]["routes"] = ["knowledge", "knowledge"]
        with self.assertRaisesRegex(EvaluationError, "duplicate route"):
            evaluate(dataset, run, digest, "hash")
        run["samples"][0]["routes"] = ["unexpected"]
        with self.assertRaisesRegex(EvaluationError, "unknown route"):
            evaluate(dataset, run, digest, "hash")

    def test_invalid_reviews_and_human_label_provenance(self):
        dataset, run, digest = fixture()
        run["samples"][0]["reviews"].append(copy.deepcopy(run["samples"][0]["reviews"][0]))
        with self.assertRaisesRegex(EvaluationError, "duplicate reviewer"):
            evaluate(dataset, run, digest, "hash")
        dataset["label_status"] = "human_reviewed"
        with self.assertRaisesRegex(EvaluationError, "label_review"):
            validate_dataset(dataset)
        dataset["label_review"] = "Reviewed by test reviewer for testing"
        validate_dataset(dataset)

    def test_cli_error_is_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.semantic", "--dataset", str(DATASET), "--run", "missing.json"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Evaluation error:", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_cli_produces_deterministic_report(self):
        _, run, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(run), encoding="utf-8")
            command = [sys.executable, "-m", "evaluation.semantic", "--dataset", str(DATASET), "--run", str(path)]
            first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(json.loads(first.stdout)["routing"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
