# Retrieval evaluation

This evaluates retrieval only, not generated answers. `holdout.v1.json` is a
small versioned set of six English/Spanish queries and binary positive
judgments at the **embedded document ID** level. Each judgment identifies an
exact source URL, heading and short supporting excerpt. No product-name alias
is assumed. The holdout was constructed from existing local Getnet embeddings;
one Spanish query was additionally checked against a corpus-wide keyword
search after inspecting the exact-baseline results. This pooling can favor the
baseline, so this is an initial diagnostic set, not an unbiased benchmark or
a basis for a production quality claim.

The actual source corpus is generated and git-ignored; it is **not** contained
in this repository. The holdout records the SHA-256 of both complete embeddings
JSONL files used for labeling. IDs can change when source text or chunking
changes. Keep those exact snapshots in a controlled local location. If they
are unavailable, or their fingerprints no longer match, do not claim a new
corpus is the same benchmark: review and version new judgments and rankings.
The source URLs may also change after extraction. Do not copy scraped content
into the repository to recreate the snapshots.

Judgments are intentionally incomplete: unlisted documents are **unjudged**,
not asserted irrelevant. In metric calculation unjudged results contribute
zero gain; the report also counts them so that they can be reviewed. Scores
are against the known relevant documents, not necessarily every relevant
document in the corpus. Duplicate/near-duplicate passages count separately.
Ask a domain reviewer to decide ambiguous candidates rather than guessing;
preserve source evidence and increment the holdout version when judgments,
queries or corpus snapshots change. Do not tune a retriever on this set and
then call the resulting score an independent holdout result; use a new
independently judged set for that.

## Metrics

For each query, at rank cutoff `k`:

* Recall@k is the number of unique labeled relevant IDs retrieved in the top
  k divided by the total labeled relevant IDs for that query.
* MRR@k is the reciprocal rank of the first labeled relevant ID in the top k,
  or zero; the report macro-averages it over queries.
* nDCG@k uses binary gain 1 for labeled relevant IDs and 0 otherwise, discount
  `1 / log2(rank + 1)`, and an ideal ranking of up to k relevant IDs.

The report includes per-query scores and macro averages, plus counts of judged
relevant and unjudged results. Empty result lists are allowed, while missing
queries, duplicate IDs, invalid depths, mismatched versions and fingerprints
are errors. No arbitrary pass/fail quality threshold is imposed.

## Offline baseline and audit

The checked-in `exact-cosine.v1.json` is a frozen ranking over both snapshot
corpora, using all 1,649 stored embeddings, CPU E5 query embeddings with
`query: `, float32 vectors, exact cosine similarity, and ID ordering for exact
ties. Query-model revision is
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. The snapshots contain stored
embedding model names but do not record their model revision, so matching
query and passage weights cannot be independently established from those
files. This baseline does not include Redis HNSW approximation or the
production tool's Get Classic keyword lookup and reranking; it is not a
measurement of the current tool.

Score the frozen run with only Python's standard library:

```bash
python -m retrieval_evaluation.evaluate score \
  --rankings retrieval_evaluation/exact-cosine.v1.json -k 5
python -m unittest -v retrieval_evaluation.test_evaluate
```

Audit labels and fingerprints against the exact local snapshots:

```bash
python -m retrieval_evaluation.evaluate audit \
  --en /path/to/getnet_en_embeddings.jsonl \
  --mx /path/to/getnet_mx_embeddings.jsonl
```

To regenerate the exact baseline, install the repository's existing dependencies
and make the specified model revision available in the local Hugging Face cache
beforehand. The command forces CPU and `local_files_only=True`; it does not
download the model. Numeric differences between software/hardware versions can
change near ties, so save the generated ranking for reproducible scoring:

```bash
python -m retrieval_evaluation.evaluate capture-exact \
  --en /path/to/getnet_en_embeddings.jsonl \
  --mx /path/to/getnet_mx_embeddings.jsonl --depth 5 > exact.json
```

## Compare the current tool

Use an **isolated, read-only evaluation Redis** containing exactly the two
snapshots; stop all index writers during verification and capture. Configure
`GETNET_REDIS_URL` and `[knowledge].vector_index` as for the application.
Confirm the target before running: this command reads the entire configured
index and document hashes, but never writes to Redis. It verifies source
fingerprints, all indexed IDs, metadata and float32 vectors before calling
the actual `vector_search` tool, including its Get Classic special case. It
does not call an LLM or the external search tool. The current tool's normal
E5 model loader may download its model if it is not cached; pre-cache it to
avoid network access. The current tool does not pin a model revision: record
the environment and verify the loaded model matches the baseline before
interpreting the comparison.

```bash
python -m retrieval_evaluation.evaluate capture-current \
  --en /path/to/getnet_en_embeddings.jsonl \
  --mx /path/to/getnet_mx_embeddings.jsonl --depth 5 > current.json
python -m retrieval_evaluation.evaluate score --rankings current.json \
  --baseline retrieval_evaluation/exact-cosine.v1.json -k 5
```

Output includes both reports and current-minus-baseline macro deltas. Capture
depth is the actual tool's requested result count (maximum 10); scoring at a
smaller k slices that capture, which is not necessarily identical to calling
the tool directly with that smaller count. Redis HNSW and the current tool's
model/runtime are not guaranteed deterministic across index builds or
environments. Repeat captures when assessing variability; scoring the same
saved ranking and holdout is deterministic. A comparison is meaningful only
with matching corpus, query model and depth, with the limitations above.

CI runs only offline unit/integration tests with synthetic rankings and mocked
retrieval; it does not connect to Redis, load embeddings, or make paid calls.
