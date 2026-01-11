"""Analyzer server for extracting themes and insights from articles."""

from typing import Any, Dict, List

from fastmcp import FastMCP
from pydantic_ai import Agent

from sotaforge.utils.constants import ANALYZER_SYSTEM_PROMPT, PYDANTIC_AI_MODEL
from sotaforge.utils.dataclasses import (
    NotParsedDocument,
    ParsedDocument,
    ThemesAndInsights,
)
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)
server = FastMCP("analyzer")
db_store = ChromaStore()

# Create Pydantic AI agent with structured output
analyzer_agent = Agent(
    PYDANTIC_AI_MODEL,
    output_type=ThemesAndInsights,
    system_prompt=ANALYZER_SYSTEM_PROMPT,
)


@server.tool(
    name="analyze_documents",
    description=(
        "Analyzes documents from a source collection, enriches with "
        "themes/insights, and stores results into a destination collection. "
        "Use to move parsed→analyzed."
    ),
)
async def analyze_documents(
    document_to_process_collection: str,
    document_processed_collection: str,
) -> Dict[str, Any]:
    """Analyze all documents in a collection and return enriched documents."""
    logger.info(
        "Retrieving documents from collection: %s",
        document_to_process_collection,
    )
    documents = db_store.fetch_documents(document_to_process_collection)

    if not documents:
        logger.warning(
            "No documents found in collection '%s'",
            document_to_process_collection,
        )
        return {
            "source_collection": document_to_process_collection,
            "results": [],
            "count": 0,
        }

    logger.info(
        "Analyzing %s documents from '%s'",
        len(documents),
        document_to_process_collection,
    )

    analyzed_docs: List[ParsedDocument] = []

    for doc in documents:
        # Ensure doc is a ParsedDocument instance
        if isinstance(doc, NotParsedDocument):
            document = ParsedDocument.from_not_parsed(doc)
        elif isinstance(doc, ParsedDocument):
            document = doc
        else:
            logger.warning(f"Skipping document of unknown type: {type(doc)}")
            continue
        logger.debug(f"Analyzing document: {document.title}")

        prompt = (
            "Analyze this document and extract key themes,"
            " trends, challenges, and opportunities.\n\n"
            f"Title: {document.title}\n"
            f"Source Type: {document.source_type}\n\n"
            # Keep prompt context short to avoid token overages
            f"Document text (truncated):\n{document.text[:1200]}"
        )

        try:
            result = await analyzer_agent.run(prompt)
            extraction = result.output

            document.themes = extraction.themes
            document.insights = extraction.insights

            logger.debug(
                "Extracted %s themes and %s insights from %s",
                len(extraction.themes),
                len(extraction.insights),
                document.title,
            )
        except Exception as e:
            logger.warning(f"Failed to analyze {document.title}: {e}")

        analyzed_docs.append(document)

    logger.info("Analysis complete: %s documents analyzed", len(analyzed_docs))

    # Store enriched documents automatically (full content) in destination collection
    db_store.upsert_documents(document_processed_collection, analyzed_docs)

    # Return trimmed payload to limit tokens sent back to the LLM
    return {
        "source_collection": document_to_process_collection,
        "destination_collection": document_processed_collection,
        "count": len(analyzed_docs),
        "stored_count": len(analyzed_docs),
        "results": [doc.to_dict_with_text_limit(800) for doc in analyzed_docs],
    }


async def launch_analyzer_server() -> None:
    """Launch the analyzer server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
