"""Pytest configuration and fixtures for SOTAforge tests.

Test Markers:
    @pytest.mark.online - Tests requiring internet connectivity
    @pytest.mark.slow - Tests that take a long time to run
    @pytest.mark.integration - Integration tests

Usage:
    Run all tests:
        pytest

    Skip online tests:
        pytest -m "not online"

    Skip slow and online tests (used in pre-commit):
        pytest -m "not online and not slow"

    Run only integration tests:
        pytest -m integration
"""

from typing import Any, Dict

import pytest


@pytest.fixture
def sample_not_parsed_document() -> Dict[str, Any]:
    """Return sample NotParsedDocument data."""
    return {
        "title": "Sample Research Paper",
        "url": "https://arxiv.org/abs/2024.12345",
        "source_type": "paper",
        "snippet": "This is a snippet",
        "abstract": "This is a sample abstract about AI research.",
        "authors": ["John Doe", "Jane Smith"],
        "year": 2024,
        "venue": "NeurIPS",
        "metadata": {"category": "cs.AI"},
    }


@pytest.fixture
def sample_parsed_document() -> Dict[str, Any]:
    """Return sample ParsedDocument data."""
    return {
        "title": "Sample Research Paper",
        "url": "https://arxiv.org/abs/2024.12345",
        "source_type": "paper",
        "snippet": "This is a snippet",
        "abstract": "This is a sample abstract about AI research.",
        "authors": ["John Doe", "Jane Smith"],
        "year": 2024,
        "venue": "NeurIPS",
        "text": "Full text of the research paper goes here.",
        "themes": ["AI", "Machine Learning"],
        "insights": ["Novel approach to training", "Better performance"],
        "metadata": {"category": "cs.AI"},
    }


@pytest.fixture
def sample_web_result() -> Dict[str, Any]:
    """Return sample web search result data."""
    return {
        "title": "AI News Article",
        "url": "https://example.com/ai-article",
        "source_type": "web",
        "snippet": "Breaking news about AI developments.",
        "abstract": "",
        "authors": [],
        "year": 0,
        "venue": "",
        "metadata": {},
    }


@pytest.fixture
def mock_fastmcp_server() -> Any:
    """Return a mock FastMCP server for testing."""

    class MockTool:
        def __init__(
            self,
            name: str,
            description: str = "",
            parameters: Dict[str, Any] | None = None,
        ):
            self.name = name
            self.description = description
            self.parameters = parameters or {"type": "object", "properties": {}}

    class MockServer:
        async def get_tools(self) -> Dict[str, MockTool]:
            return {
                "search.query": MockTool(
                    "search.query",
                    "Search for documents",
                    {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
                "parse.document": MockTool(
                    "parse.document", "Parse a document", {"type": "object"}
                ),
                "internal.debug": MockTool("internal.debug", "Debug tool"),
            }

    return MockServer()
