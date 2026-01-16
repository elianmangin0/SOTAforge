"""Constants for the SOTAforge project."""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv(".env.secrets")
# LLM Configuration
MODEL: str = "gpt-5-nano"
PDF_PARSING_MAX_TOKENS = 65535  # Maximum tokens for PDF text extraction
PDF_PARSING_TEMPERATURE = 0.1  # Temperature for PDF parsing
MAX_ORCHESTRATOR_MESSAGES = 80

# Search constants
SERPER_URL = "https://google.serper.dev/search"
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 3
CHROMA_PATH = "data/chroma"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "SOTAforge API"
API_DESCRIPTION = "REST API for generating State-of-the-Art research summaries"
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Next.js default dev server
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

# Pipeline tuning constants
MAX_MESSAGE_HISTORY = 30  # Trim message history to prevent token overages
DOC_TEXT_LIMIT_FULL = 3000  # Character limit for full document text storage
DOC_TEXT_LIMIT_RETURN = 800  # Character limit for document text in LLM responses
ANALYZER_PROMPT_TEXT_LIMIT = 10_000  # Character limit for analyzer prompt context
SYNTHESIZER_PROMPT_TEXT_LIMIT = 5_000  # Character limit for synthesizer prompt context
MAX_RETRIES = 3  # Maximum validation retries per pipeline step

# Request timeouts (in seconds)
REQUEST_TIMEOUT_WEB = 10  # Timeout for fetching web pages
REQUEST_TIMEOUT_PDF = 30  # Timeout for fetching PDF files

# Rate limiting
MAX_CONCURRENT_PARSING_REQUESTS = 3  # Limit concurrent PDF/web parsing requests
MAX_CONCURRENT_PDF_PAGES = 5  # Limit concurrent PDF page parsing

# PDF parsing limits
MAX_PARSED_PDF_PAGES = 10


class CollectionNames(Enum):
    """ChromaDB collection names used throughout the pipeline."""

    RAW = "raw"
    FILTERED = "filtered"
    PARSED = "parsed"
    ANALYZED = "analyzed"
    SYNTHESIZED = "synthesized"
    FINAL_SOTA = "8_final_sota"
