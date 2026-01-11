"""Combined search server for web and papers."""

import json
import os
from typing import Any, Dict, List, Union

import feedparser
import requests
from fastmcp import FastMCP

from sotaforge.utils.constants import ARXIV_API, MAX_RESULTS, SERPER_URL
from sotaforge.utils.dataclasses import NotParsedDocument, SourceType
from sotaforge.utils.errors import ConfigurationError, SearchError
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)
server = FastMCP("search")


@server.tool(
    name="search_web",
    description=(
        "Searches the web for HTML pages on a given topic using Google Serper API. "
        "Returns an object containing the original 'query' and 'results'."
    ),
)
async def search_web(
    query: str, max_results: int = MAX_RESULTS
) -> Dict[str, Union[str, List[dict[str, Any]]]]:
    """Search the web for pages based on the query.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Dict with 'query' and 'results' keys. 'results' is a list of Document objects.

    Raises:
        SearchError: If query is empty or max_results is invalid
        ConfigurationError: If SERPER_API_KEY is not set

    """
    # Validate inputs
    if not query or not query.strip():
        raise SearchError("Search query cannot be empty")

    if max_results < 1 or max_results > 100:
        raise SearchError("max_results must be between 1 and 100")

    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        raise ConfigurationError("SERPER_API_KEY environment variable is not set")

    logger.info(f"Searching web for: {query}")

    # Calculate number of pages needed (10 results per page)
    num_pages = (max_results + 9) // 10

    results: List[NotParsedDocument] = []

    for page in range(1, num_pages + 1):
        logger.debug(f"Fetching page {page} for query: {query}")

        payload = json.dumps(
            {
                "q": query,
                "page": page,
            }
        )

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(SERPER_URL, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()

            for item in data.get("organic", []):
                if len(results) >= max_results:
                    break

                results.append(
                    NotParsedDocument(
                        title=item.get("title", "No title"),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", "No snippet available"),
                        source_type=SourceType.WEB,
                    )
                )
                logger.debug(f"Found: {item.get('link', '')}")

        except Exception as e:
            logger.exception(f"Error during search for '{query}' page {page}: {e}")
            continue

    logger.info(f"Collected {len(results)} web results for query: {query}")
    return {"query": query, "results": [doc.to_dict() for doc in results]}


@server.tool(
    name="search_papers",
    description=(
        "Searches arXiv for papers on a given topic. "
        "Returns an object containing the original 'query' and 'results'."
    ),
)
async def search_papers(
    query: str, max_results: int = MAX_RESULTS
) -> Dict[str, Union[str, List[dict[str, Any]]]]:
    """Search arXiv for papers based on the query.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        Dict with 'query' and 'results' keys.
        'results' is a list of NotParsedDocument objects converted to dicts.

    Raises:
        SearchError: If query is empty or max_results is invalid

    """
    # Validate inputs
    if not query or not query.strip():
        raise SearchError("Search query cannot be empty")

    if max_results < 1 or max_results > 100:
        raise SearchError("max_results must be between 1 and 100")

    logger.info(f"Searching arXiv for query: {query}")

    # arXiv API: search in all fields, order by relevance
    url = (
        f"{ARXIV_API}?search_query=all:{query.replace(' ', '+')}"
        f"&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    )

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning(f"Failed to query arXiv: {e}")
        return {"query": query, "results": []}

    results: List[NotParsedDocument] = []

    for entry in feed.entries[:max_results]:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        link = entry.get("link", "")

        # Authors
        authors = [a.get("name", "").strip() for a in entry.get("authors", [])]

        # Year from published date
        published = entry.get("published", "")
        year = 0
        if published:
            try:
                year = int(published[:4])
            except ValueError:
                year = 0

        # Venue: try journal_ref or primary_category
        venue = ""
        if "arxiv_journal_ref" in entry:
            venue = entry.get("arxiv_journal_ref", "")
        elif "arxiv_primary_category" in entry:
            cat = entry.get("arxiv_primary_category", {})
            venue = cat.get("term", "") if isinstance(cat, dict) else str(cat)

        results.append(
            NotParsedDocument(
                title=title or f"Untitled arXiv paper {len(results) + 1}",
                authors=authors,
                year=year,
                venue=venue,
                abstract=summary,
                url=link,
                source_type=SourceType.PAPER,
            )
        )
        logger.debug(f"Found: {link}")

    logger.info(f"Collected {len(results)} arXiv papers for query: {query}")
    return {"query": query, "results": [doc.to_dict() for doc in results]}


async def launch_search_server() -> None:
    """Launch the search server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
