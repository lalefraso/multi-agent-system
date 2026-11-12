# Getnet content ingestion

The miner collects section-level text from Getnet's English (`/en`) or Mexico (`/mx`) sitemap. Requests and redirects are restricted to the selected locale on `www.getnet.net`; URLs disallowed by `robots.txt` stop the run. Stale sitemap pages returning 404/410 are reported and skipped, and content in the wrong language is excluded. For Mexico, the miner also discovers same-locale links on the homepage so current articles omitted from its sitemap are included.

Each heading and its associated text is stored as one record; the miner does not split sections into chunks. The `section_heading` is a hierarchical path formed from the page title and the heading/subheading. Visually grouped carousel cards are extracted as individual subsections. Paragraphs, list items, and table rows remain separate semantic blocks. Articles, client stories, legal pages, and product/business pages receive separate category values.

## Run

From the repository root:

```bash
.venv/bin/python -m pip install -r getnet_ingestion/requirements.txt
.venv/bin/python -m getnet_ingestion.miner --locale en
.venv/bin/python -m getnet_ingestion.miner --locale mx
```

The generated corpora are `getnet_ingestion/data/getnet_en_sections.jsonl` and `getnet_ingestion/data/getnet_mx_sections.jsonl`. They are git-ignored because they contain fetched source content. Use `--output` to choose another output path.

The command fails explicitly if the sitemap, robots policy, page fetch, redirect, HTML type, extraction, or output write fails.

## Vectorize

Both locales use `intfloat/multilingual-e5-small` and normalized 384-dimensional vectors. Text is embedded with the model-required `passage: ` prefix while the stored source text remains unchanged. The model's input length is checked before encoding; long sections are separated at paragraph, list-item, or table-row boundaries preserved by the miner. An individual paragraph that remains too long is grouped at sentence boundaries. A sentence that still exceeds the model limit stops the run instead of being truncated.

```bash
.venv/bin/python -m getnet_ingestion.embedder --locale en
.venv/bin/python -m getnet_ingestion.embedder --locale mx
```

The resulting `getnet_<locale>_embeddings.jsonl` files are also git-ignored. Retrieval queries must use the same model with the `query: ` prefix.

## Persist in Redis

Redis persistence stores each embedding as a hash under the shared `getnet:section:<id>` namespace and creates the `idx:getnet` Redis Search HNSW index using cosine distance. English and Mexico records remain distinguishable through their `language` field. Synthetic customer JSON records use the `customers:customer:<customer_id>` key namespace. These namespaces and the vector index are the same in local and production Redis; only the connection URL changes. In Redis browser tools the prefixes appear as logical groups, not actual folders. Redis Stack or a Redis service with the Search/vector module enabled is required.

```bash
.venv/bin/python -m getnet_ingestion.redis_store --locale all --recreate-index
```

By default, the command reads both locale embeddings files and loads the Redis connection URL from `GETNET_REDIS_URL` or `REDIS_URL` in the repository-root `.env` file. Use `--locale en` or `--locale mx` to upsert only one corpus without recreating the shared index. Alternatively, pass `--redis-url`. There is no localhost fallback. Keep credentials in `.env`, which must not be committed.

`--recreate-index` requires both locale corpora, removes the previous `idx:getnet:en` and `idx:getnet:mx` indexes and their documents, then rebuilds the shared `idx:getnet` index with both corpora. Without it, records are upserted and old records absent from the input remain. Run mining and vectorization for both locales successfully before recreating the shared index.

The store validates the embedding model, dimensions, finite vector values, and unique record IDs before writing.

### Initialize knowledge and synthetic customers

Use the same initializer for local or production Redis. It upserts both locale corpora into `idx:getnet` and adds five synthetic customer records without overwriting existing customer keys:

```bash
.venv/bin/python -m getnet_ingestion.seed
```

Set `GETNET_REDIS_URL` (or `REDIS_URL`) in the environment or repository-root `.env` to select the target. For a one-off target, pass `--redis-url`. Confirm the target URL before running this command against a production deployment. In Docker Compose, the initializer is run automatically against the local Redis service.

To deliberately replace the existing Getnet index and all documents indexed by it, add `--recreate-index`. The command still preserves the customer keys.
