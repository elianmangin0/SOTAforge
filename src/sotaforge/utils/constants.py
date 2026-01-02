"""Constants for the SOTAforge project."""

MODEL: str = "gpt-4.1-nano"
PYDANTIC_AI_MODEL: str = "openai:gpt-4.1-nano"
SERPER_URL = "https://google.serper.dev/search"
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 10
MAX_ORCHESTRATOR_MESSAGES = 80
CHROMA_PATH = "data/chroma"

# Pipeline tuning constants
MAX_MESSAGE_HISTORY = 30  # Trim message history to prevent token overages
DOC_TEXT_LIMIT_FULL = 3000  # Character limit for full document text storage
DOC_TEXT_LIMIT_RETURN = 800  # Character limit for document text in LLM responses
ANALYZER_PROMPT_TEXT_LIMIT = 1200  # Character limit for analyzer prompt context
MAX_RETRIES = 3  # Maximum validation retries per pipeline step


# Collection names
class CollectionNames:
    """ChromaDB collection names used throughout the pipeline."""

    RAW = "raw"
    FILTERED = "filtered"
    PARSED = "parsed"
    ANALYZED = "analyzed"
    SYNTHESIZED = "synthesized"
    FINAL_SOTA = "8_final_sota"


ANALYZER_SYSTEM_PROMPT = """
You are an expert research analyst specializing in extracting 
key insights from academic and technical documents.

Your task is to analyze documents and identify:
- Key trends and emerging patterns
- Technical challenges and limitations
- Research opportunities and future directions
- Methodological approaches
- Important findings and contributions

Provide clear, concise themes formatted as "Category: Description" where 
category is one of: Trend, Challenge, Opportunity, Method, Finding.
"""

SYNTHESIZER_SYSTEM_PROMPT = """
You are an expert technical writer specializing in state-of-the-art reviews.
"""

SYNTHESIZER_PROMPT = """
You are an expert technical writer specializing in state-of-the-art reviews. 
Generate a comprehensive, well-structured SOTA document.

Structure your response with these sections:

## 1. Introduction
Brief context and significance of the topic.

## 2. Core Technologies & Approaches
Key technologies, methodologies, and architectural approaches currently used.

## 3. Best Practices
Industry and research best practices, design patterns, and recommendations.

## 4. Challenges & Limitations
Current challenges, bottlenecks, and known limitations in the field.

## 5. Emerging Trends
Recent trends, innovations, and cutting-edge developments.

## 6. Research Opportunities
Open problems and promising research directions.

## 7. Sources & References
Key sources, papers, and resources that informed this analysis.
Use the following format for references:
- [Title](URL) - Brief description of the source.

## 8. Conclusion
Summary and outlook for the field.

Format the response in Markdown. 
Include specific technologies, frameworks, and tools mentioned in the source materials.

Here are the analyzed documents to base your synthesis on:
"""
