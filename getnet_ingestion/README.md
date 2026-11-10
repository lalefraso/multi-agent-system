# Getnet English content mining

This step collects section-level text from pages listed in Getnet's English sitemap. It does not follow page links, so only sitemap entries on `www.getnet.net` whose path is `/en` or starts with `/en/` are requested. Redirects outside that scope are rejected, URLs disallowed by `robots.txt` stop the run, stale sitemap pages returning 404/410 are reported and skipped, and non-English page content is detected and excluded.

Each heading and its associated text is stored as one record; the miner does not split sections into chunks. The `section_heading` is a hierarchical path formed from the page title and the heading/subheading, for example `Agentic Commerce - From eCommerce to Agentic Commerce` or `Agentic Commerce - Our Agentic Commerce Suite - MCP AI Tool Kit`. Visually grouped carousel cards are also extracted as individual subsections under their section heading. The `text` field begins with that heading path, followed by the section body. Paragraphs, list items, and table rows remain separate semantic blocks in the extracted text. Articles, client stories, legal pages, and product/business pages receive separate category values. Language detection uses the combined text of each page, falling back to the English sitemap locale only when the page text is too short or ambiguous.

## Run

From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r getnet_ingestion/requirements.txt
.venv/bin/python -m getnet_ingestion.miner
```

The generated JSONL corpus is written to `getnet_ingestion/data/getnet_en_sections.jsonl` and is intentionally git-ignored because it is fetched source content. To choose another output path:

```bash
.venv/bin/python -m getnet_ingestion.miner --output /path/to/corpus.jsonl
```

The command fails explicitly if the sitemap, robots policy, page fetch, redirect, HTML type, extraction, or output write fails.

## Vectorize

Use the open-source `BAAI/bge-small-en-v1.5` model to create normalized 384-dimensional vectors. Its 512-token input limit accommodates more complete subsections than `all-MiniLM-L6-v2` (256 tokens):

```bash
.venv/bin/python -m getnet_ingestion.embedder
```

The output is `getnet_ingestion/data/getnet_en_embeddings.jsonl`. Each JSONL row retains the source record and adds `embedding_model`, `embedding_dimensions`, and `embedding`. This generated file is git-ignored along with the source corpus.

Before encoding, the command tokenizes every section without truncation and checks it against the model's actual maximum sequence length. If a section is too long, the vectorizer separates it only at the paragraph, list-item, or table-row boundaries preserved by the miner. Each resulting record retains the same semantic `section_heading`; no arbitrary token/sentence chunks or chunk-index metadata are added. If a single semantic block is still too long, vectorization stops and reports it rather than truncating or splitting that block. BGE is English-focused. For retrieval, use the same BGE model to embed queries and consider the model card's recommended query instruction. To choose another input or output:

```bash
.venv/bin/python -m getnet_ingestion.embedder \
  --input getnet_ingestion/data/getnet_en_sections.jsonl \
  --output getnet_ingestion/data/getnet_en_embeddings.jsonl
```
