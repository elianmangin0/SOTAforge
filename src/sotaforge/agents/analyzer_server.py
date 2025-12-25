"""Analyzer server for extracting themes and insights from articles."""

from typing import Any, Dict, List

from fastmcp import FastMCP
from pydantic_ai import Agent

from sotaforge.utils.constants import ANALYZER_SYSTEM_PROMPT, PYDANTIC_AI_MODEL
from sotaforge.utils.dataclasses import Document, NotParsedDocument, ThemesAndInsights
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
        "Analyzes documents from a collection, adding themes and insights. "
        "Retrieves parsed documents from the collection, runs analysis, "
        "and returns updated documents."
    ),
)
async def analyze_documents(collection: str) -> Dict[str, Any]:
    """Analyze all documents in a collection and return enriched documents."""
    logger.info(f"Retrieving documents from collection: {collection}")
    documents = db_store.fetch_documents(collection)

    if not documents:
        logger.warning(f"No documents found in collection '{collection}'")
        return {"collection": collection, "results": [], "count": 0}

    logger.info(f"Analyzing {len(documents)} documents from '{collection}'")

    analyzed_docs: List[Document] = []

    for doc in documents:
        # Ensure doc is a Document instance
        if isinstance(doc, NotParsedDocument):
            document = Document.from_not_parsed(doc)
        elif isinstance(doc, Document):
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
            f"Document text:\n{document.text[:3000]}"
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
    return {
        "collection": collection,
        "count": len(analyzed_docs),
        "results": [doc.to_dict() for doc in analyzed_docs],
    }


async def launch_analyzer_server() -> None:
    """Launch the analyzer server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
