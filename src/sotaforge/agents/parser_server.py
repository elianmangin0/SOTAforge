"""Parser server for extracting and parsing documents from search results."""

from typing import Any, Dict

from fastmcp import FastMCP

from sotaforge.utils.dataclasses import Document, NotParsedDocument
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.logger import get_logger
from sotaforge.utils.parsing import parse_paper_result, parse_web_result

logger = get_logger(__name__)
server = FastMCP("parser")
db_store = ChromaStore()


@server.tool(
    name="parse_documents",
    description=(
        "Parses documents from a collection to extract full text content. "
        "Retrieves NotParsedDocuments from the collection, parses them, "
        "and returns Documents with text."
    ),
)
async def parse_documents(
    collection: str,
) -> Dict[str, Any]:
    """Parse all documents from a collection and extract text content.

    Args:
        collection: The collection name to retrieve and parse from

    Returns:
        Dict with collection name, count, and parsed documents

    """
    # Step 1: Retrieve documents from collection
    logger.info(f"Retrieving documents from collection: {collection}")
    documents = db_store.fetch_documents(collection)

    if not documents:
        logger.warning(f"No documents found in collection '{collection}'")
        return {
            "collection": collection,
            "parsed_documents": [],
            "count": 0,
        }

    logger.info(f"Retrieved {len(documents)} documents from '{collection}'")

    # Step 2: Parse each document
    parsed_docs = []
    for doc in documents:
        # Convert to NotParsedDocument if needed
        if isinstance(doc, NotParsedDocument):
            not_parsed = doc
        elif isinstance(doc, Document):
            # Already parsed, skip or convert to NotParsedDocument for re-parsing
            logger.debug(
                f"Document '{doc.title}' already has text : {doc.text}, skipping parse"
            )
            parsed_docs.append(doc)
            continue
        else:
            doc_dict = doc if isinstance(doc, dict) else doc.to_dict()
            not_parsed = NotParsedDocument.from_dict(doc_dict)

        try:
            # Use appropriate parser based on source type
            if not_parsed.source_type == "paper":
                parsed_doc = await parse_paper_result(not_parsed)
            else:  # web
                parsed_doc = await parse_web_result(not_parsed)

            parsed_docs.append(parsed_doc)
            logger.debug(f"Parsed: {not_parsed.url}")
        except Exception as e:
            logger.warning(f"Failed to parse {not_parsed.url}: {e}")
            # Keep document even if parsing fails, just without text
            parsed_docs.append(
                Document.from_not_parsed(not_parsed, text="Failed to parse content.")
            )

    logger.info(f"Successfully parsed {len(parsed_docs)}/{len(documents)} documents")
    return {
        "results": [doc.to_dict() for doc in parsed_docs],
    }


async def launch_parser_server() -> None:
    """Launch the parser server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
