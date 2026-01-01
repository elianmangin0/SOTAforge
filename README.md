# SOTAforge 🚀

> **Portfolio Project**: An intelligent research pipeline that generates comprehensive State-of-the-Art summaries using AI agents and real-time vector storage.

## Project Overview

SOTAforge is a full-stack application that automates the research synthesis process. Given any topic, it orchestrates a multi-stage pipeline to search, filter, analyze, and synthesize information into a coherent SOTA (State-of-the-Art) summary. Built to demonstrate modern full-stack architecture and AI integration.

---

## 🛠️ Technologies & Architecture

### Backend
- **FastAPI** - High-performance Python API with streaming responses (SSE)
- **OpenAI API** - GPT-4 for LLM-powered validation and synthesis
- **FastMCP** - Model Context Protocol for modular agent architecture
- **ChromaDB** - Vector database for intelligent document storage and retrieval
- **Pydantic AI** - Structured LLM outputs for reliable data extraction
- **AsyncIO** - Concurrent processing throughout the pipeline

**Key Features:**
- Server-Sent Events (SSE) for real-time progress streaming
- Automatic message history trimming to prevent token overages
- Error resilience with intelligent rate-limit handling
- Modular agent design (search, filter, parse, analyze, synthesize)

### Frontend
- **Next.js 14** - Modern React framework with TypeScript
- **EventSource API** - Real-time progress updates from backend
- **Tailwind CSS** - Beautiful, responsive UI
- **Terminal-style Log Display** - Auto-scrolling, real-time progress visualization

### Data Pipeline
```
Search → Filter → Parse → Analyze → Synthesize
  ↓        ↓        ↓        ↓         ↓
 Raw   Filtered  Parsed  Analyzed  SOTA Summary
(ChromaDB Collections)
```

Each stage automatically stores results in ChromaDB for the next stage to consume.

---

## 💡 The Idea

**Problem**: Staying updated on research requires reading dozens of papers/articles—time-consuming and error-prone.

**Solution**: Automate the synthesis:
1. **Search**: Web + academic paper sources for the topic
2. **Filter**: Score & keep only high-quality, relevant documents
3. **Parse**: Extract full text from PDFs and web pages
4. **Analyze**: Use AI to identify themes, trends, challenges, opportunities
5. **Synthesize**: Generate a polished SOTA report

All orchestrated by an LLM that validates each step with automatic retries.

**Result**: Comprehensive research summaries in minutes, not hours.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+, Node.js 18+
- OpenAI API key
- Serper API key (web/paper search)

### Backend
```bash
pip install -e .  # or: uv sync
export OPENAI_API_KEY=sk-...
export SERPER_API_KEY=...
python -m sotaforge.api
# http://localhost:8000
```

### Frontend
```bash
cd frontend && npm install
echo "NEXT_PUBLIC_SOTAFORGE_API_URL=http://localhost:8000" > .env.local
npm run dev
# http://localhost:3000
```

---

## 🎯 Key Features

✅ Real-time Progress Streaming (watch pipeline execute step-by-step)  
✅ LLM-Orchestrated Pipeline (AI validates each step with retries)  
✅ Vector Database Integration (intelligent document storage/retrieval)  
✅ Modular Agent Architecture (each stage is independent MCP server)  
✅ Token-Aware Processing (auto-trims message history)  
✅ Production-Ready Error Handling (graceful failures, detailed logging)  

---

## 📈 What This Demonstrates

- Full-stack integration with real-time SSE streaming
- AI/LLM orchestration with validation loops
- Vector database practical usage
- Async Python with high concurrency
- Modern frontend with real-time updates
- Production patterns: rate-limiting, error recovery, optimization
- Clean API design (REST + streaming)

---

## 📝 API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/sota` | POST | `{ "topic": "..." }` - Start SOTA generation |
| `/api/sota/stream/{task_id}` | GET | SSE stream of real-time progress |

---

## 📄 License

Portfolio project. Feel free to use as reference or extend.

**Built with** ❤️ to showcase modern AI + full-stack development.
