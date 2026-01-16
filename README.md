# SOTAforge

An intelligent research pipeline that generates comprehensive State-of-the-Art summaries using AI agents and real-time storage.

## Project Overview

SOTAforge is a full-stack application that automates the research synthesis process. Given any topic, it orchestrates a multi-stage pipeline to search, filter, analyze, and synthesize information into a coherent SOTA (State-of-the-Art) summary. Built to demonstrate modern full-stack architecture and AI integration.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Demo](#demo)
- [Technologies & Architecture](#technologies--architecture)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Data Pipeline](#data-pipeline)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Backend](#backend-1)
  - [Frontend](#frontend-1)

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

Parsing step has been edited out since it is a bit long.

Test it yourself [here](https://sotaforge-front.onrender.com/)


## Technologies & Architecture

### Backend
- **FastAPI** - High-performance Python API with streaming responses
- **OpenAI API** - GPT-5 for LLM-powered validation and synthesis
- **FastMCP** - Model Context Protocol for modular agent architecture
- **ChromaDB** - Database for document storage and retrieval
- **Pydantic AI** - Structured LLM outputs for reliable data extraction
- **AsyncIO** - Concurrent processing throughout the pipeline

**Key Features:**
- Server-Sent Events (SSE) for real-time progress streaming
- Email delivery with PDF and Markdown attachments
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
- Python 3.12+, Node.js 18+
- OpenAI API key
- Serper API key
- Resend API key (for email notifications)

### Backend
```bash
uv sync
uv run api
# http://localhost:8000
```

Create a `.env.secrets` file based on `.env.secrets.template` and add your API keys and Resend configuration:
```bash
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
RESEND_API_KEY=re_your_resend_api_key
SENDER_EMAIL=your_mail@your_domain
```

**Note:** Sign up at [Resend](https://resend.com) to get your API key. Free tier includes 100 emails/day and 3,000 emails/month.

**Email Delivery:** Once generation completes, users receive an email with:
- PDF report (professionally formatted)
- Markdown file (raw SOTA content)

### Frontend
```bash
cd frontend && npm install
echo "NEXT_PUBLIC_SOTAFORGE_API_URL=http://localhost:8000" > .env.local
npm run dev
# http://localhost:3000
```



