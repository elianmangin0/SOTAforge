"""Tests for `search_server`."""

import os
from unittest.mock import MagicMock, patch

import pytest

from sotaforge.utils.errors import ConfigurationError, SearchError


@pytest.mark.asyncio
async def test_search_web_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful web search."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "organic": [
            {
                "title": "Result 1",
                "link": "https://example.com/1",
                "snippet": "Snippet 1",
            },
            {
                "title": "Result 2",
                "link": "https://example.com/2",
                "snippet": "Snippet 2",
            },
        ]
    }
    mock_response.raise_for_status = MagicMock()

    # Mock requests.post
    with patch("requests.post", return_value=mock_response):
        monkeypatch.setenv("SERPER_API_KEY", "test_api_key")

        func = search_server.search_web.fn
        result = await func("test query", 2)

        assert result["query"] == "test query"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Result 1"
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][0]["source_type"] == "web"
        assert result["results"][1]["title"] == "Result 2"


@pytest.mark.asyncio
async def test_search_web_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test web search with empty query raises error."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    monkeypatch.setenv("SERPER_API_KEY", "test_api_key")

    func = search_server.search_web.fn

    with pytest.raises(SearchError, match="Search query cannot be empty"):
        await func("", 10)

    with pytest.raises(SearchError, match="Search query cannot be empty"):
        await func("   ", 10)


@pytest.mark.asyncio
async def test_search_web_invalid_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test web search with invalid max_results raises error."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    monkeypatch.setenv("SERPER_API_KEY", "test_api_key")

    func = search_server.search_web.fn

    with pytest.raises(SearchError, match="max_results must be between 1 and 100"):
        await func("query", 0)

    with pytest.raises(SearchError, match="max_results must be between 1 and 100"):
        await func("query", 101)


@pytest.mark.asyncio
async def test_search_web_missing_api_key() -> None:
    """Test web search without API key raises error."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Ensure API key is not set
    old_key = os.environ.pop("SERPER_API_KEY", None)
    try:
        func = search_server.search_web.fn

        with pytest.raises(
            ConfigurationError, match="SERPER_API_KEY environment variable is not set"
        ):
            await func("query", 10)
    finally:
        # Restore key if it existed
        if old_key:
            os.environ["SERPER_API_KEY"] = old_key


@pytest.mark.asyncio
async def test_search_web_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test web search handles pagination correctly."""
    import importlib
    from typing import Any

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock multiple pages
    call_count = 0

    def mock_post(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": f"Result {i}",
                    "link": f"https://example.com/{i}",
                    "snippet": f"Snippet {i}",
                }
                for i in range((call_count - 1) * 10 + 1, call_count * 10 + 1)
            ]
        }
        mock_response.raise_for_status = MagicMock()
        return mock_response

    with patch("requests.post", side_effect=mock_post):
        monkeypatch.setenv("SERPER_API_KEY", "test_api_key")

        func = search_server.search_web.fn
        result = await func("test query", 25)

        assert result["query"] == "test query"
        # Should make 3 calls (10 + 10 + 5 = 25)
        assert call_count == 3
        # Should return exactly 25 results
        assert len(result["results"]) == 25


@pytest.mark.asyncio
async def test_search_web_api_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test web search handles API errors gracefully."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock failing request
    with patch("requests.post", side_effect=Exception("API Error")):
        monkeypatch.setenv("SERPER_API_KEY", "test_api_key")

        func = search_server.search_web.fn
        result = await func("test query", 10)

        # Should return empty results instead of crashing
        assert result["query"] == "test query"
        assert result["results"] == []


@pytest.mark.asyncio
async def test_search_papers_success() -> None:
    """Test successful arXiv paper search."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock feedparser response
    mock_feed = MagicMock()
    mock_feed.entries = [
        {
            "title": "Paper 1",
            "summary": "Abstract 1",
            "link": "https://arxiv.org/abs/2024.1234",
            "authors": [{"name": "Author 1"}, {"name": "Author 2"}],
            "published": "2024-01-15",
            "arxiv_journal_ref": "Conference 2024",
        },
        {
            "title": "Paper 2",
            "summary": "Abstract 2",
            "link": "https://arxiv.org/abs/2024.5678",
            "authors": [{"name": "Author 3"}],
            "published": "2024-02-20",
            "arxiv_primary_category": {"term": "cs.AI"},
        },
    ]

    with patch("feedparser.parse", return_value=mock_feed):
        func = search_server.search_papers.fn
        result = await func("machine learning", 2)

        assert result["query"] == "machine learning"
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Paper 1"
        assert result["results"][0]["authors"] == ["Author 1", "Author 2"]
        assert result["results"][0]["year"] == 2024
        assert result["results"][0]["venue"] == "Conference 2024"
        assert result["results"][0]["source_type"] == "paper"
        assert result["results"][1]["venue"] == "cs.AI"


@pytest.mark.asyncio
async def test_search_papers_empty_query() -> None:
    """Test paper search with empty query raises error."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    func = search_server.search_papers.fn

    with pytest.raises(SearchError, match="Search query cannot be empty"):
        await func("", 10)

    with pytest.raises(SearchError, match="Search query cannot be empty"):
        await func("   ", 10)


@pytest.mark.asyncio
async def test_search_papers_invalid_max_results() -> None:
    """Test paper search with invalid max_results raises error."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    func = search_server.search_papers.fn

    with pytest.raises(SearchError, match="max_results must be between 1 and 100"):
        await func("query", 0)

    with pytest.raises(SearchError, match="max_results must be between 1 and 100"):
        await func("query", 101)


@pytest.mark.asyncio
async def test_search_papers_parsing_error() -> None:
    """Test paper search handles parsing errors gracefully."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock feedparser to raise exception
    with patch("feedparser.parse", side_effect=Exception("Parse error")):
        func = search_server.search_papers.fn
        result = await func("test query", 10)

        # Should return empty results instead of crashing
        assert result["query"] == "test query"
        assert result["results"] == []


@pytest.mark.asyncio
async def test_search_papers_missing_fields() -> None:
    """Test paper search handles missing/incomplete entry fields."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock feedparser with incomplete data
    mock_feed = MagicMock()
    mock_feed.entries = [
        {
            # Missing title, authors, venue
            "summary": "Abstract",
            "link": "https://arxiv.org/abs/2024.1234",
            "published": "invalid-date",  # Invalid date format
        }
    ]

    with patch("feedparser.parse", return_value=mock_feed):
        func = search_server.search_papers.fn
        result = await func("query", 1)

        assert len(result["results"]) == 1
        # Should have fallback title
        assert "Untitled arXiv paper" in result["results"][0]["title"]
        assert result["results"][0]["authors"] == []
        assert result["results"][0]["year"] == 0
        assert result["results"][0]["venue"] == ""


@pytest.mark.asyncio
async def test_search_papers_max_results_limit() -> None:
    """Test that search_papers respects max_results limit."""
    import importlib

    search_server = importlib.import_module("sotaforge.agents.search_server")

    # Mock feedparser with many entries
    mock_feed = MagicMock()
    mock_feed.entries = [
        {
            "title": f"Paper {i}",
            "summary": f"Abstract {i}",
            "link": f"https://arxiv.org/abs/2024.{i:04d}",
            "authors": [],
            "published": "2024-01-01",
        }
        for i in range(50)
    ]

    with patch("feedparser.parse", return_value=mock_feed):
        func = search_server.search_papers.fn
        result = await func("query", 10)

        # Should only return max_results items
        assert len(result["results"]) == 10
