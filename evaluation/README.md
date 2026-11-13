# Semantic evaluation, v1

This is an offline **scoring harness**, not evidence that the system performs
well. [`cases/v1.json`](cases/v1.json) contains the ten prompts in the challenge,
with routing labels proposed from the current router prompt, not human-approved
gold labels. Weather and exchange-rate routing is deliberately unlabeled because
general web-search behavior is unresolved. The user identifiers in the
challenge are not included; use a test principal and synthetic data when
executing cases. No observations, human judgments, or results are supplied.

## What is measured

- Routing is exact **set** match against `expected_routes`: `knowledge`,
  `customer_support`, and/or `human_escalation`. Empty sets represent no
  delegation. Order and repeated calls are not measured; unlabeled cases are
  excluded and counted separately. Labels are hypotheses, not a policy gate.
- A human reviewer judges the whole answer on four dimensions: groundedness
  (claims supported by the evidence actually available to the system), factual
  correctness (claims consistent with independently verified evidence),
  relevance (addresses the request), and usefulness (actionable and appropriately
  qualified for the request). Use `pass`, `fail`, or `not_applicable`, with a
  rationale for every dimension. Do not equate correctness with grounding: an
  unsupported true claim can be ungrounded. For a refusal or a response without
  factual claims, mark a dimension not applicable only with an explanation;
  unsupported factual claims should fail groundedness, not become N/A.
- A pass on groundedness requires an exact quote from an `agent_context`
  source; factual correctness requires one from a `reference` source. Include
  citations and rationale that cover **all**
  material claims, not merely one convenient fact. Source provenance matters:
  recorded tool outputs establish what was available, while correctness also
  needs an independently verified reference. The harness checks quote
  membership only; it cannot verify entailment, completeness, source
  trustworthiness, reviewer independence, or whether the evidence reached the
  model. A failed grounding judgment may have no source at all.
- A second independent human review is optional. The report computes raw
  pairwise agreement per dimension, excluding N/A pairs, and uses only the
  first review for quality rates. Review independently before sharing judgments;
  investigate disagreements and document adjudication outside this report.
  Raw agreement is not a calibrated judge or a measure of validity.

Rates are per sample, not per unique case; repeated samples weight that case
more heavily. Each metric reports its denominator and N/A count. Unreviewed
samples do not contribute quality scores; missing cases and unlabeled routing
are reported. No score is emitted for an empty denominator (`null`). There are
no acceptance thresholds, hallucination-rate estimates, statistical claims,
automatic semantic judge, or quality regression gate. A passing unit test or a
score against proposed labels is **not** a real-model quality result.

## Collecting an actual run

1. Review and version the dataset and expected routes with a domain/product
   owner before calling it gold. Change `dataset_id` when content or labels
   change, set `label_status` to `human_reviewed`, and put reviewer identity,
   date and review reference in `label_review`. Leave unresolved labels `null`.
   Review proposed labels against current behavior and policy; do not assume
   the challenge's questions contain their own answers.
2. Separately run the cases against a specified system revision and model,
   with authorized test identities and a controlled, documented evidence
   snapshot. Record each response's `answer` and `agents_used` from
   `POST /chat` as `routes`, and the evidence available to the answering agent
    from an authorized trace or controlled tool outputs. The API response alone
   does not expose those sources and is insufficient for grounding review.
   Record independent factual references as additional evidence, identifying
   their origin and marking them `reference`; mark actual tool output
   `agent_context`. Do not invent evidence or include secrets or customer data.
   Record the execution procedure, settings, dependencies, data snapshot,
   model/version, sampling settings, and run time externally and reference them
   in `system.configuration`. For repeats use a distinct `sample_id` per case;
   preserve each actual output rather than selecting favorable ones. Record
   failures separately and disclose them: this v1 format scores completed
   nonempty answers only, not service failure rates.
3. Have reviewers inspect the question, answer, and sources; populate reviews
   with exact source quotes. An empty `reviews` array is valid for pending
   review. Keep raw runs and traces in an appropriately access-controlled
   location; do not commit sensitive content.
4. Compute the SHA-256 of the exact dataset bytes, e.g.
   `shasum -a 256 evaluation/cases/v1.json`, and place it in the run file.
   Score with:

   ```bash
   python -m evaluation.semantic --dataset evaluation/cases/v1.json --run path/to/run.json > path/to/report.json
   ```

The report includes dataset and run SHA-256 hashes, run metadata, case coverage,
per-sample route matches and verdicts, aggregate counts/rates and agreement.
Identical input bytes produce identical output; changing the dataset invalidates
the recorded hash. Retain the exact versioned inputs and the report together.
Invalid inputs fail with a nonzero exit; the tool never calls a model or network.

## Run format

The following is a **format illustration only**, not a recorded result. Replace
every placeholder, including the dataset hash; omit reviews until actually
performed. `origin` identifies the source. Fields are mandatory and unknown fields are rejected.

```json
{
  "schema_version": 1,
  "dataset_id": "challenge-scenarios-v1",
  "dataset_sha256": "<sha256 of exact dataset bytes>",
  "run_id": "<unique run id>",
  "system": {
    "revision": "<git revision>",
    "model": "<provider and model version>",
    "configuration": "<reference to execution and evidence snapshot>"
  },
  "samples": [
    {
      "case_id": "product-comparison",
      "sample_id": "1",
      "answer": "<actual answer>",
      "routes": ["knowledge"],
      "evidence": [
        {"id": "source-1", "kind": "agent_context", "origin": "<actual tool output provenance>", "text": "<actual tool output>"},
        {"id": "source-2", "kind": "reference", "origin": "<independent reference provenance>", "text": "<actual reference text>"}
      ],
      "reviews": [
        {
          "reviewer": "<reviewer identifier>",
          "dimensions": {
            "groundedness": {"verdict": "pass", "rationale": "<why>", "citations": [{"evidence_id": "source-1", "quote": "<exact substring of actual tool output>"}]},
            "factual_correctness": {"verdict": "pass", "rationale": "<why and independent verification>", "citations": [{"evidence_id": "source-2", "quote": "<exact substring of actual reference text>"}]},
            "relevance": {"verdict": "pass", "rationale": "<why>", "citations": []},
            "usefulness": {"verdict": "pass", "rationale": "<why>", "citations": []}
          }
        }
      ]
    }
  ]
}
```

In an actual review, include separate source entries and citations for tool
evidence and independent verification, as appropriate. For `fail` and
`not_applicable`, citations may be empty; explain the judgment. The exact
`routes` vocabulary is the API's `agents_used` vocabulary, not tool names.

## CI

The app unittest suite exercises the harness using fabricated in-memory inputs
only. It does not execute the agents, call an LLM, judge semantic quality, or
write a quality report. Human-reviewed data, live runs, and decisions about
acceptance thresholds remain separate work.
