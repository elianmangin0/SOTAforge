# SOTAforge

An intelligent research pipeline that generates comprehensive State-of-the-Art summaries using AI agents and real-time storage.

## Project Overview

SOTAforge is a full-stack application that automates the research synthesis process. Given any topic, it orchestrates a multi-stage pipeline to search, filter, analyze, and synthesize information into a coherent SOTA (State-of-the-Art) summary. Built to demonstrate modern full-stack architecture and AI integration.


## Problem Statement

Staying updated on research requires reading dozens of papers and articles, a process that is time-consuming and error-prone. SOTAforge automates the synthesis through a multi-stage pipeline:

1. **Search**: Gather web and academic paper sources for the topic
2. **Filter**: Score and retain only high-quality, relevant documents
3. **Parse**: Extract full text from PDFs and web pages
4. **Analyze**: Use AI to identify themes, trends, challenges, and opportunities
5. **Synthesize**: Generate a polished SOTA report

An LLM orchestrator validates each step with automatic retries, delivering comprehensive research summaries in minutes.

## Demo

https://github.com/user-attachments/assets/49106f4c-3f99-4f4e-887d-4b74f9a4783e

Parsing step has been edited out since it is a bit long with docling on cpu.


## Technologies & Architecture

### Backend
- **FastAPI** - High-performance Python API with streaming responses
- **OpenAI API** - GPT-4 for LLM-powered validation and synthesis
- **FastMCP** - Model Context Protocol for modular agent architecture
- **ChromaDB** - Database for document storage and retrieval
- **Pydantic AI** - Structured LLM outputs for reliable data extraction
- **AsyncIO** - Concurrent processing throughout the pipeline

**Key Features:**
- Server-Sent Events (SSE) for real-time progress streaming
- Automatic message history trimming to prevent token overages
- Error resilience with intelligent rate-limit handling
- Modular agent design (search, filter, parse, analyze, synthesize)

### Frontend
- **Next.js 14**
- **EventSource API**
- **Tailwind CSS**
- **Terminal-style Log Display**




### Data Pipeline
```
Search → Filter → Parse → Analyze → Synthesize
  ↓        ↓        ↓        ↓         ↓
 Raw   Filtered  Parsed  Analyzed  SOTA Summary
```

Each stage automatically stores results in ChromaDB for the next stage to consume.

## Quick Start

### Prerequisites
- Python 3.10+, Node.js 18+
- OpenAI API key
- Serper API key

### Backend
```bash
uv sync
export OPENAI_API_KEY=sk-...
export SERPER_API_KEY=...
uv run api
# http://localhost:8000
```

### Frontend
```bash
cd frontend && npm install
echo "NEXT_PUBLIC_SOTAFORGE_API_URL=http://localhost:8000" > .env.local
npm run dev
# http://localhost:3000
```

## Next Steps

Potential enhancements to take this project further:

### Production Readiness
- Deploy backend with Docker and orchestration
- Add authentication and rate limiting for API endpoints
- Implement distributed task queue for background processing
- Set up monitoring and observability
- Add comprehensive test coverage and CI/CD pipelines

### Enhanced Search Coverage
- Generate multiple search queries from the base topic using LLM query expansion
- Implement semantic query diversification to capture different angles
- Add cross-language search capabilities
- Support domain-specific search engines (arXiv, PubMed, IEEE Xplore)

### Improved Parsing Performance
- Deploy GPU-accelerated document parsing with specialized models or use Vision Language Models (VLM) via API for quicker and better parsing
- Implement parallel parsing with distributed workers
- Add support for complex document formats (equations, diagrams, charts)


