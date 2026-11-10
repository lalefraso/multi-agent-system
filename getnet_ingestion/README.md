# Getnet English content mining

This step collects section-level text from pages listed in Getnet's English sitemap. It does not follow page links, so only sitemap entries on `www.getnet.net` whose path is `/en` or starts with `/en/` are requested. Redirects outside that scope are rejected, URLs disallowed by `robots.txt` stop the run, stale sitemap pages returning 404/410 are reported and skipped, and non-English page content is detected and excluded.

Each heading and its associated text is stored as one record; the miner does not split sections into chunks. The `text` field contains the page title, section heading, and section body on separate lines, with no labels. Records also include a stable content ID, source URL, page title, section heading, language, category, and extraction timestamp. Articles, client stories, legal pages, and product/business pages receive separate category values. Language detection uses the combined text of each page, falling back to the English sitemap locale only when the page text is too short or ambiguous.

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
