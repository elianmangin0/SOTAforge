"""Tests for `analyzer_server`."""

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_analyze_documents_calls_upsert_and_sets_insights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that analyze_documents calls upsert and sets insights."""
    # Ensure llm client creation won't fail when importing analyzer_server
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")
    from sotaforge.utils.models import ParsedDocument

    doc = ParsedDocument(title="A Doc", text="Important content.")

    upsert_called: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> None:
        upsert_called["collection"] = collection
        upsert_called["docs"] = docs

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: [doc], upsert_documents=fake_upsert),
    )

    async def fake_run(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            output=SimpleNamespace(themes=["theme1"], insights=["insight1"])
        )

    # Replace the whole agent with a test double that exposes `run`
    monkeypatch.setattr(
        analyzer_server,
        "analyzer_agent",
        SimpleNamespace(run=fake_run),
    )

    func = analyzer_server.analyze_documents.fn
    res = await func("src_col", "dst_col")

    assert res["count"] == 1
    assert res["stored_count"] == 1
    assert res["destination_collection"] == "dst_col"
    assert res["results"][0]["title"] == "A Doc"
    assert upsert_called["collection"] == "dst_col"


@pytest.mark.asyncio
async def test_analyze_documents_empty_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test analyzing when source collection is empty."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: []),
    )

    func = analyzer_server.analyze_documents.fn
    res = await func("empty_col", "dst_col")

    assert res["count"] == 0
    assert res["results"] == []
    assert res["source_collection"] == "empty_col"


@pytest.mark.asyncio
async def test_analyze_documents_with_not_parsed_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test analyzing converts NotParsedDocument to ParsedDocument."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")
    from sotaforge.utils.models import NotParsedDocument

    not_parsed = NotParsedDocument(title="Not Parsed", url="http://example.com")

    upsert_called = {}

    def fake_upsert(collection: str, docs: list[Any]) -> None:
        upsert_called["docs"] = docs

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(
            fetch_documents=lambda c: [not_parsed], upsert_documents=fake_upsert
        ),
    )

    async def fake_run(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(
            output=SimpleNamespace(themes=["theme"], insights=["insight"])
        )

    monkeypatch.setattr(
        analyzer_server, "analyzer_agent", SimpleNamespace(run=fake_run)
    )

    func = analyzer_server.analyze_documents.fn
    res = await func("src", "dst")

    assert res["count"] == 1
    # Check that the document was converted and has themes/insights
    assert len(upsert_called["docs"]) == 1
    stored_doc = upsert_called["docs"][0]
    assert stored_doc.title == "Not Parsed"
    assert stored_doc.themes == ["theme"]
    assert stored_doc.insights == ["insight"]


@pytest.mark.asyncio
async def test_analyze_documents_handles_analysis_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that analysis failures are logged but don't stop processing."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")
    from sotaforge.utils.models import ParsedDocument

    doc = ParsedDocument(title="Doc", text="Content")

    upsert_called = {}

    def fake_upsert(collection: str, docs: list[Any]) -> None:
        upsert_called["docs"] = docs

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: [doc], upsert_documents=fake_upsert),
    )

    async def fake_run_error(prompt: str) -> None:
        raise Exception("Analysis failed")

    monkeypatch.setattr(
        analyzer_server, "analyzer_agent", SimpleNamespace(run=fake_run_error)
    )

    func = analyzer_server.analyze_documents.fn
    res = await func("src", "dst")

    # Document should still be stored, just without themes/insights
    assert res["count"] == 1
    assert len(upsert_called["docs"]) == 1
    stored_doc = upsert_called["docs"][0]
    assert stored_doc.title == "Doc"


@pytest.mark.asyncio
async def test_analyze_documents_multiple_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test analyzing multiple documents."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")
    from sotaforge.utils.models import ParsedDocument

    docs = [
        ParsedDocument(title="Doc1", text="Content 1"),
        ParsedDocument(title="Doc2", text="Content 2"),
        ParsedDocument(title="Doc3", text="Content 3"),
    ]

    upsert_called: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> None:
        upsert_called["collection"] = collection
        upsert_called["docs"] = docs

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs, upsert_documents=fake_upsert),
    )

    call_count = {"count": 0}

    async def fake_run(prompt: str) -> SimpleNamespace:
        call_count["count"] += 1
        return SimpleNamespace(
            output=SimpleNamespace(
                themes=[f"theme_{call_count['count']}"],
                insights=[f"insight_{call_count['count']}"],
            )
        )

    monkeypatch.setattr(
        analyzer_server, "analyzer_agent", SimpleNamespace(run=fake_run)
    )

    func = analyzer_server.analyze_documents.fn
    res = await func("src", "dst")

    assert res["count"] == 3
    assert res["stored_count"] == 3
    assert len(res["results"]) == 3
    # Agent should have been called 3 times (once per doc)
    assert call_count["count"] == 3
    # All docs should be stored
    assert len(upsert_called["docs"]) == 3


@pytest.mark.asyncio
async def test_analyze_documents_skips_unknown_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that documents of unknown type are skipped."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")
    from sotaforge.utils.models import ParsedDocument

    docs = [
        ParsedDocument(title="Valid", text="Content"),
        {"not": "a document object"},  # Invalid type
    ]

    upsert_called = {}

    def fake_upsert(collection: str, docs: list[Any]) -> None:
        upsert_called["docs"] = docs

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs, upsert_documents=fake_upsert),
    )

    async def fake_run(prompt: str) -> SimpleNamespace:
        return SimpleNamespace(output=SimpleNamespace(themes=[], insights=[]))

    monkeypatch.setattr(
        analyzer_server, "analyzer_agent", SimpleNamespace(run=fake_run)
    )

    func = analyzer_server.analyze_documents.fn
    res = await func("src", "dst")

    # Only 1 valid document should be processed
    assert res["count"] == 1
    assert len(upsert_called["docs"]) == 1


@pytest.mark.asyncio
async def test_analyze_documents_truncates_long_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that long document text is truncated in the prompt."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    analyzer_server = importlib.import_module("sotaforge.agents.analyzer_server")
    from sotaforge.utils.models import ParsedDocument

    # Create a document with very long text
    long_text = "x" * 100000
    doc = ParsedDocument(title="Long Doc", text=long_text)

    monkeypatch.setattr(
        analyzer_server,
        "db_store",
        SimpleNamespace(
            fetch_documents=lambda c: [doc], upsert_documents=lambda c, d: None
        ),
    )

    prompts_received = []

    async def fake_run(prompt: str) -> SimpleNamespace:
        prompts_received.append(prompt)
        return SimpleNamespace(output=SimpleNamespace(themes=[], insights=[]))

    monkeypatch.setattr(
        analyzer_server, "analyzer_agent", SimpleNamespace(run=fake_run)
    )

    func = analyzer_server.analyze_documents.fn
    await func("src", "dst")

    # Check that the prompt doesn't contain the full text
    assert len(prompts_received) == 1
    prompt = prompts_received[0]
    # The prompt should be much shorter than the original text
    assert len(prompt) < len(long_text)
    # And it should contain truncated text indicator
    from sotaforge.utils.constants import ANALYZER_PROMPT_TEXT_LIMIT

    assert str(long_text[:ANALYZER_PROMPT_TEXT_LIMIT]) in prompt
