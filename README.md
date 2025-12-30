# SOTAforge

DB-backed research pipeline that combines modular agents with ChromaDB storage.

## Monorepo Structure
- Backend (Python): core agents and API under [src/sotaforge](src/sotaforge).
- Frontend (Next.js + TS): simple UI under [frontend](frontend).

The frontend calls the backend `POST /api/sota` endpoint to generate State-of-the-Art summaries for a given topic.

## Quick start
- Install: `pip install -e .` (or `uv sync` if you use uv).
- Run orchestrator server: `uv run server`.
- Run interactive client: `uv run client` and enter a topic.

### Backend API (FastAPI)
- Start: `python -m sotaforge.api`
- Default URL: `http://localhost:8000`
- Required env vars: `OPENAI_API_KEY`, `SERPER_API_KEY`
- Health: `GET /` and `GET /health`
- SOTA endpoint: `POST /api/sota` with body `{ "topic": "your subject" }`

### Frontend (Next.js)
Located in [frontend](frontend). Simple page with the app name, explanation, a GitHub link, and a centered topic input that calls the backend and displays the result.

Run it:

```bash
cd frontend
cp .env.example .env.local  # set NEXT_PUBLIC_SOTAFORGE_API_URL if not localhost:8000
npm install
npm run dev
# Visit http://localhost:3000
```

Notes:
- CORS is enabled in the backend for `http://localhost:3000`.
- If your backend runs on a different host/port, set `NEXT_PUBLIC_SOTAFORGE_API_URL` in `.env.local`.

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
