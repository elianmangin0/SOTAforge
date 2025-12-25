# SOTAforge

DB-backed research pipeline that combines modular agents with ChromaDB storage.

## Quick start
- Install: `pip install -e .` (or `uv sync` if you use uv).
- Run orchestrator server: `uv run server`.
- Run interactive client: `uv run client` and enter a topic.

## Pipeline (DB-first)
- `pipeline_search(topic, run_id?)` -> stores web & paper results and returns a `run_id` plus handles.
- `pipeline_filter(run_id, topic)` -> filters stored results and writes filtered handles.
- `pipeline_parse(run_id)` -> parses filtered items and stores documents.
- `pipeline_analyze(run_id)` -> analyzes parsed docs and stores themes/insights.
- `pipeline_synthesize(run_id)` -> writes the final SOTA text (also stored).

Always reuse the `run_id` returned by `pipeline_search`. Do not pass raw documents between steps; rely on the stored handles.

## DB agent tools
- `db_store_records(collection, items)`
- `db_fetch_records(collection, ids=None)`
- `db_query_records(collection, query, n_results=5)`
- `db_delete_records(collection, ids)`
- `db_reset_collection(collection)`

Chroma data is stored under `data/chroma` by default. Override with `SOTAFORGE_CHROMA_PATH` if you need a different location.
