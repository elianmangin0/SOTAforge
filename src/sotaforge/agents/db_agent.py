"""Database agent using ChromaDB for storage."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from sotaforge.utils.dataclasses import Document, NotParsedDocument, ParsedDocument
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.errors import DatabaseError
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)
server = FastMCP("db")
store = ChromaStore()


def _parse_documents_from_dict(
    items: List[Dict[str, Any]],
) -> list[Document]:
    """Parse a list of dicts into ParsedDocument or NotParsedDocument instances."""
    if not all(isinstance(item, dict) for item in items):
        raise DatabaseError("All items must be dictionaries")
    if all("text" in item and isinstance(item["text"], str) for item in items):
        return [ParsedDocument.from_dict(item) for item in items]
    elif all("text" not in item for item in items):
        return [NotParsedDocument.from_dict(item) for item in items]
    else:
        raise DatabaseError(
            "The list provided must contain either"
            " all ParsedDocument items with 'text' field"
            " or all NotParsedDocument items without 'text' field."
        )


@server.tool(
    name="store_records",
    description=(
        "Store one or more documents in a ChromaDB collection. Each record should"
        " include at least 'title' and optionally other fields like 'url', 'text',"
        " 'source_type', etc."
    ),
)
async def store_records(collection: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store or update documents in a ChromaDB collection."""
    ids = store.upsert_documents(collection, _parse_documents_from_dict(items))
    return {"collection": collection, "count": len(ids), "ids": ids}


@server.tool(
    name="fetch_documents",
    description=(
        "Fetch all documents from a ChromaDB collection. Returns a list of "
        "document dicts containing all stored fields (title, url, snippet, "
        "abstract, text, etc.)."
    ),
)
async def fetch_documents(collection: str, limit: int | None = None) -> Dict[str, Any]:
    """Fetch documents from a collection.

    Args:
        collection: The collection name to fetch from
        limit: Optional limit on number of documents to retrieve

    Returns:
        Dict with collection name, count, and list of documents as dicts

    """
    documents = store.fetch_documents(collection, limit=limit)
    return {
        "collection": collection,
        "count": len(documents),
        "documents": [doc.to_dict() for doc in documents],
    }


@server.tool(
    name="store_tool_results",
    description=(
        "Store the results of previous tool calls by providing their tool_call_ids. "
        "The system will automatically extract the tool results from the conversation "
        "and store them in the specified collection. Provide the collection name and "
        "a list of tool_call_ids to store."
    ),
)
async def store_tool_results(
    collection: str,
    tool_call_ids: List[str],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Store tool results from conversation history by tool_call_id.

    Args:
        collection: The ChromaDB collection to store results in
        tool_call_ids: List of tool_call_ids whose results should be stored
        messages: The conversation message history (injected by orchestrator)

    Returns:
        Dict with collection name, count of stored items, and their IDs

    """
    msgs = messages
    if not msgs:
        return {"error": "No messages provided", "collection": collection, "count": 0}

    # Extract tool results from messages
    items_to_store: List[Dict[str, Any]] = []
    logger.info(f"Storing tool results for tool_call_ids: {tool_call_ids}")
    logger.debug(f"Total messages to scan: {messages=}")
    logger.debug(
        f"Type of messages: {type(messages)=}[{type(msgs[0]) if msgs else 'N/A'}]"
    )

    for tool_call_id in tool_call_ids:
        # Find the tool response with matching tool_call_id
        for msg in msgs:
            if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                try:
                    result_data = msg.get("content")
                    logger.debug(f"Processing tool result for {tool_call_id}")

                    # Attempt to parse if it's a string
                    if isinstance(result_data, str):
                        try:
                            result_data = json.loads(result_data)
                            logger.debug("Parsed result_data as JSON")
                        except json.JSONDecodeError:
                            try:
                                result_data = ast.literal_eval(result_data)
                                logger.debug(
                                    "Parsed result_data using ast.literal_eval"
                                )
                            except (ValueError, SyntaxError) as e:
                                logger.error(
                                    f"Failed to parse result_data string for "
                                    f"{tool_call_id}: {e}"
                                )
                                raise DatabaseError(
                                    f"Unable to parse tool result for {tool_call_id}"
                                ) from e

                    logger.debug(
                        "Parsed result_data type=%s keys=%s",
                        type(result_data),
                        list(result_data.keys())
                        if isinstance(result_data, dict)
                        else None,
                    )

                    # Unwrap if there's a "result" field (orchestrator wrapper)
                    if isinstance(result_data, dict) and "result" in result_data:
                        result_data = result_data["result"]
                        logger.debug("Unwrapped result_data to 'result' field")

                    logger.debug(f"Result payload parsed for {tool_call_id}")
                    logger.debug(f"Result data : {result_data=}")
                    # Now we have either a dict or a list of dicts
                    # If it's a dict with "results", extract that list
                    if isinstance(result_data, dict) and "results" in result_data:
                        results = result_data["results"]
                        if isinstance(results, list):
                            items_to_store.extend(
                                [r for r in results if isinstance(r, dict)]
                            )
                        elif isinstance(results, dict):
                            items_to_store.append(results)
                    # If it's already a list of dicts, use it directly
                    elif isinstance(result_data, list):
                        items_to_store.extend(
                            [r for r in result_data if isinstance(r, dict)]
                        )
                    # If it's a single dict, wrap it
                    elif isinstance(result_data, dict):
                        items_to_store.append(result_data)
                    else:
                        logger.warning(
                            f"Unexpected result format for {tool_call_id}: "
                            f"{type(result_data)}"
                        )

                except Exception as e:
                    logger.error(f"Failed to parse tool result for {tool_call_id}: {e}")
                    continue
                break

    if not items_to_store:
        logger.warning("No valid items found to store from the provided tool_call_ids")
        return {
            "collection": collection,
            "count": 0,
            "message": "No valid items found from the provided tool_call_ids",
        }

    # Normalize and store the items
    logger.info(f"Items to store: {len(items_to_store)}")
    logger.debug(
        f"First item sample: {items_to_store[0] if items_to_store else 'None'}"
    )

    logger.debug(f"Storing documents into collection: {collection}")
    ids = store.upsert_documents(collection, _parse_documents_from_dict(items_to_store))

    logger.info(
        f"Successfully stored {len(ids)} documents in collection '{collection}'"
    )

    return {
        "collection": collection,
        "count": len(ids),
        "ids": ids,
        "message": (
            f"Successfully stored {len(ids)} items from {len(tool_call_ids)} tool calls"
        ),
    }


async def launch_db_server() -> None:
    """Launch the DB agent server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
