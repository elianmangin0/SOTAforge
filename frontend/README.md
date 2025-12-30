# SOTAforge Frontend (in monorepo)

This is the Next.js + TypeScript frontend co-located in the SOTAforge repo.

## What this does
- Landing page with app name, explanation, and GitHub link.
- Topic input form that calls backend `POST /api/sota`.
- Displays backend JSON result.

## Requirements
- Node.js 18+
- Backend running on `http://localhost:8000` with `OPENAI_API_KEY` and `SERPER_API_KEY` set.

## Run
```bash
# from SOTAforge/frontend
cp .env.example .env.local
npm install
npm run dev
# Visit http://localhost:3000
```

Set `NEXT_PUBLIC_SOTAFORGE_API_URL` in `.env.local` if your backend runs elsewhere.
