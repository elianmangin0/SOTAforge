"""Tests for `parser_server`."""

from typing import Any

import pytest


class FakeChromaStore:
    """Fake ChromaStore for testing parser functionality."""

    def __init__(self, documents: list[Any] | None = None):
        """Initialize fake store with documents."""
        self._documents = documents or []

    def fetch_documents(self, collection: str) -> list[Any]:
        """Return stored documents."""
        return self._documents

    def upsert_documents(self, collection: str, documents: list[Any]) -> None:
        """Store documents (no-op for testing)."""
        pass


async def fake_parse_paper_result(not_parsed: Any) -> Any:
    """Fake parser for paper documents."""
    from sotaforge.utils.models import ParsedDocument

    return ParsedDocument.from_not_parsed(
        not_parsed,
        text=f"Parsed text for {not_parsed.title}",
        themes=["AI", "Research"],
        insights=["Novel approach", "Good results"],
    )


async def fake_parse_web_result(not_parsed: Any) -> Any:
    """Fake parser for web documents."""
    from sotaforge.utils.models import ParsedDocument

    return ParsedDocument.from_not_parsed(
        not_parsed,
        text=f"Parsed web content for {not_parsed.title}",
        themes=["Technology"],
        insights=["Interesting development"],
    )


@pytest.mark.asyncio
async def test_parse_documents_empty_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test parsing when source collection is empty."""
    import importlib

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    # Mock empty collection
    fake_store = FakeChromaStore(documents=[])
    monkeypatch.setattr(parser_server, "db_store", fake_store)

    func = parser_server.parse_documents.fn
    result = await func("source_collection", "dest_collection")

    assert result["source_collection"] == "source_collection"
    assert result["count"] == 0
    assert result["parsed_documents"] == []


@pytest.mark.asyncio
async def test_parse_documents_paper_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing paper documents."""
    import importlib

    from sotaforge.utils.models import NotParsedDocument, SourceType

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    # Create sample paper documents
    docs = [
        NotParsedDocument(
            title="Paper 1",
            url="https://arxiv.org/abs/1234.5678",
            source_type=SourceType.PAPER,
            abstract="Abstract 1",
            authors=["Author 1"],
            year=2024,
        ),
        NotParsedDocument(
            title="Paper 2",
            url="https://arxiv.org/abs/2345.6789",
            source_type=SourceType.PAPER,
            abstract="Abstract 2",
            authors=["Author 2"],
            year=2024,
        ),
    ]

    fake_store = FakeChromaStore(documents=docs)
    monkeypatch.setattr(parser_server, "db_store", fake_store)
    monkeypatch.setattr(parser_server, "parse_paper_result", fake_parse_paper_result)

    func = parser_server.parse_documents.fn
    result = await func("filtered", "parsed")

    assert result["source_collection"] == "filtered"
    assert result["destination_collection"] == "parsed"
    assert result["stored_count"] == 2
    assert len(result["results"]) == 2
    assert all("text" in doc for doc in result["results"])


@pytest.mark.asyncio
async def test_parse_documents_web_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing web documents."""
    import importlib

    from sotaforge.utils.models import NotParsedDocument, SourceType

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    # Create sample web documents
    docs = [
        NotParsedDocument(
            title="Web Article 1",
            url="https://example.com/article1",
            source_type=SourceType.WEB,
            snippet="Article snippet 1",
        ),
        NotParsedDocument(
            title="Web Article 2",
            url="https://example.com/article2",
            source_type=SourceType.WEB,
            snippet="Article snippet 2",
        ),
    ]

    fake_store = FakeChromaStore(documents=docs)
    monkeypatch.setattr(parser_server, "db_store", fake_store)
    monkeypatch.setattr(parser_server, "parse_web_result", fake_parse_web_result)

    func = parser_server.parse_documents.fn
    result = await func("filtered", "parsed")

    assert result["source_collection"] == "filtered"
    assert result["destination_collection"] == "parsed"
    assert result["stored_count"] == 2
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_parse_documents_mixed_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing mixed document types (papers and web)."""
    import importlib

    from sotaforge.utils.models import NotParsedDocument, SourceType

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    # Create mixed documents
    docs = [
        NotParsedDocument(
            title="Paper",
            url="https://arxiv.org/abs/1234.5678",
            source_type=SourceType.PAPER,
            abstract="Paper abstract",
        ),
        NotParsedDocument(
            title="Web Article",
            url="https://example.com/article",
            source_type=SourceType.WEB,
            snippet="Web snippet",
        ),
    ]

    fake_store = FakeChromaStore(documents=docs)
    monkeypatch.setattr(parser_server, "db_store", fake_store)
    monkeypatch.setattr(parser_server, "parse_paper_result", fake_parse_paper_result)
    monkeypatch.setattr(parser_server, "parse_web_result", fake_parse_web_result)

    func = parser_server.parse_documents.fn
    result = await func("filtered", "parsed")

    assert result["stored_count"] == 2
    assert len(result["results"]) == 2


@pytest.mark.asyncio
async def test_parse_documents_with_dict_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test parsing when documents are dicts instead of NotParsedDocument objects."""
    import importlib

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    # Create documents as dicts
    docs = [
        {
            "title": "Paper 1",
            "url": "https://arxiv.org/abs/1234.5678",
            "source_type": "paper",
            "abstract": "Abstract",
            "authors": ["Author"],
            "year": 2024,
            "venue": "Conference",
        }
    ]

    fake_store = FakeChromaStore(documents=docs)
    monkeypatch.setattr(parser_server, "db_store", fake_store)
    monkeypatch.setattr(parser_server, "parse_paper_result", fake_parse_paper_result)

    func = parser_server.parse_documents.fn
    result = await func("filtered", "parsed")

    assert result["stored_count"] == 1
    assert len(result["results"]) == 1


@pytest.mark.asyncio
async def test_parse_documents_handles_parsing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that parsing errors are handled gracefully."""
    import importlib

    from sotaforge.utils.models import NotParsedDocument, SourceType

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    async def failing_parser(not_parsed: Any) -> None:
        """Parser that always fails."""
        raise Exception("Parsing failed")

    docs = [
        NotParsedDocument(
            title="Paper",
            url="https://arxiv.org/abs/1234.5678",
            source_type=SourceType.PAPER,
            abstract="Abstract",
        )
    ]

    fake_store = FakeChromaStore(documents=docs)
    monkeypatch.setattr(parser_server, "db_store", fake_store)
    monkeypatch.setattr(parser_server, "parse_paper_result", failing_parser)

    func = parser_server.parse_documents.fn
    result = await func("filtered", "parsed")

    # Should still return a result, but with fallback text
    assert result["stored_count"] == 1
    assert len(result["results"]) == 1
    assert "Failed to parse content" in result["results"][0]["text"]


@pytest.mark.asyncio
async def test_parse_documents_already_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that already-parsed documents are returned as-is."""
    import importlib

    from sotaforge.utils.models import NotParsedDocument, ParsedDocument, SourceType

    parser_server = importlib.import_module("sotaforge.agents.parser_server")

    # Create a parsed document
    not_parsed = NotParsedDocument(
        title="Paper",
        url="https://arxiv.org/abs/1234.5678",
        source_type=SourceType.PAPER,
        abstract="Abstract",
    )
    docs = [
        ParsedDocument.from_not_parsed(
            not_parsed, text="Already parsed text", themes=[], insights=[]
        )
    ]

    fake_store = FakeChromaStore(documents=docs)
    monkeypatch.setattr(parser_server, "db_store", fake_store)

    func = parser_server.parse_documents.fn
    result = await func("filtered", "parsed")

    assert result["stored_count"] == 1
    assert result["results"][0]["text"] == "Already parsed text"
