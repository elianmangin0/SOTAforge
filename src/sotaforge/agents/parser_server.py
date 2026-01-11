"""Parser server for extracting and parsing documents from search results."""

import asyncio
from typing import Any, Dict

from fastmcp import FastMCP

from sotaforge.utils.constants import MAX_CONCURRENT_PARSING_REQUESTS
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.errors import ParsingError
from sotaforge.utils.logger import get_logger
from sotaforge.utils.models import NotParsedDocument, ParsedDocument
from sotaforge.utils.parsing import parse_paper_result, parse_web_result

logger = get_logger(__name__)
server = FastMCP("parser")
db_store = ChromaStore()

# Semaphore to limit concurrent document parsing
_parsing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PARSING_REQUESTS)


@server.tool(
    name="parse_documents",
    description=(
        "Parses documents from a source collection and stores parsed documents "
        "into a destination collection. Use to move filtered→parsed."
    ),
)
async def parse_documents(
    document_to_process_collection: str,
    document_processed_collection: str,
) -> Dict[str, Any]:
    """Parse all documents from a collection and extract text content.

    Args:
        document_to_process_collection: Source collection (e.g. 'filtered')
        document_processed_collection: Destination collection (e.g. 'parsed')

    Returns:
        Dict with collection names, count, and parsed documents

    """
    # Step 1: Retrieve documents from collection
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
            "parsed_documents": [],
            "count": 0,
        }

    logger.info(
        "Retrieved %s documents from '%s'",
        len(documents),
        document_to_process_collection,
    )

    # Step 2: Parse all documents in parallel
    logger.info(f"Parsing {len(documents)} documents in parallel")

    async def parse_document(doc: Any) -> ParsedDocument:
        """Parse a single document."""
        async with _parsing_semaphore:
            # Convert to NotParsedDocument if needed
            if isinstance(doc, NotParsedDocument):
                not_parsed = doc
            elif isinstance(doc, ParsedDocument):
                # Already parsed, return as-is
                logger.debug(
                    f"Document '{doc.title}' already has"
                    f" text: {doc.text}, skipping parse"
                )
                return doc
            elif isinstance(doc, dict):
                doc_dict = doc if isinstance(doc, dict) else doc.to_dict()
                not_parsed = NotParsedDocument.from_dict(doc_dict)
            else:
                raise ParsingError(f"Unsupported document type: {type(doc)}")

            try:
                # Use appropriate parser based on source type
                if not_parsed.source_type == "paper":
                    parsed_doc = await parse_paper_result(not_parsed)
                else:  # web
                    parsed_doc = await parse_web_result(not_parsed)

                logger.debug(f"Parsed: {not_parsed.url}")
                return parsed_doc
            except Exception as e:
                logger.warning(f"Failed to parse {not_parsed.url}: {e}")
                # Keep document even if parsing fails, just without text
                return ParsedDocument.from_not_parsed(
                    not_parsed, text="Failed to parse content."
                )

    # Parse all documents in parallel
    parsed_docs = await asyncio.gather(*[parse_document(doc) for doc in documents])

    logger.info(f"Successfully parsed {len(parsed_docs)}/{len(documents)} documents")

    # Store full parsed documents in destination collection
    db_store.upsert_documents(document_processed_collection, parsed_docs)

    # Return trimmed payload to reduce LLM token usage
    return {
        "source_collection": document_to_process_collection,
        "destination_collection": document_processed_collection,
        "stored_count": len(parsed_docs),
        "results": [doc.to_dict_with_text_limit(800) for doc in parsed_docs],
    }


async def launch_parser_server() -> None:
    """Launch the parser server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
