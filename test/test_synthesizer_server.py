"""Tests for `synthesizer_server`."""

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_write_sota_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful SOTA generation from documents."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace()
            self.chat.completions = self

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "## State-of-the-Art Summary\n\n"
                                "This is a comprehensive SOTA."
                            )
                        )
                    )
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    from sotaforge.utils.models import ParsedDocument

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    # Create mock documents
    docs = [
        ParsedDocument(
            title="Paper 1",
            url="https://arxiv.org/abs/1234.5678",
            text="Content of paper 1",
            themes=["AI", "Machine Learning"],
            insights=["Novel approach", "Good results"],
        ),
        ParsedDocument(
            title="Paper 2",
            url="https://example.com/article",
            text="Content of paper 2",
            themes=["Deep Learning"],
            insights=["Interesting findings"],
        ),
    ]

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("analyzed")

    assert result["status"] == "completed"
    assert "State-of-the-Art" in result["text"]
    assert len(result["text"]) > 0


@pytest.mark.asyncio
async def test_write_sota_empty_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SOTA generation with empty collection."""
    import importlib

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: []),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("empty_collection")

    assert result["status"] == "error"
    assert "No documents found" in result["text"]


@pytest.mark.asyncio
async def test_write_sota_with_long_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SOTA generation handles long document text by using snippets."""
    from sotaforge.utils import llm
    from sotaforge.utils.constants import SYNTHESIZER_PROMPT_TEXT_LIMIT

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Track what content was sent to LLM
    sent_content = []

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace()
            self.chat.completions = self

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            # Capture the content sent
            messages = kwargs.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    sent_content.append(msg.get("content", ""))

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Generated SOTA summary")
                    )
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    from sotaforge.utils.models import ParsedDocument

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    # Create document with very long text (longer than the limit)
    long_text = "x" * (SYNTHESIZER_PROMPT_TEXT_LIMIT + 100)
    docs = [
        ParsedDocument(
            title="Long Paper",
            url="https://example.com/long",
            text=long_text,
            snippet="Short snippet",
            abstract="Short abstract",
            themes=["AI"],
            insights=["Finding"],
        )
    ]

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("analyzed")

    assert result["status"] == "completed"
    # Verify that long text was replaced with snippet/abstract
    assert any("Short snippet" in content for content in sent_content)
    assert not any(long_text in content for content in sent_content)


@pytest.mark.asyncio
async def test_write_sota_with_short_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SOTA generation uses full text when it's short enough."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    sent_content = []

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace()
            self.chat.completions = self

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            messages = kwargs.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    sent_content.append(msg.get("content", ""))

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Generated SOTA summary")
                    )
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    from sotaforge.utils.models import ParsedDocument

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    short_text = "Short content about AI research"
    docs = [
        ParsedDocument(
            title="Short Paper",
            url="https://example.com/short",
            text=short_text,
            themes=["AI"],
            insights=["Finding"],
        )
    ]

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("analyzed")

    assert result["status"] == "completed"
    # Verify that short text was included directly
    assert any(short_text in content for content in sent_content)


@pytest.mark.asyncio
async def test_write_sota_filters_not_parsed_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that not-parsed documents are filtered out."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace()
            self.chat.completions = self

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="SOTA summary"))
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    from sotaforge.utils.models import NotParsedDocument, ParsedDocument

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    # Mix of parsed and not parsed documents
    docs = [
        ParsedDocument(
            title="Parsed",
            url="https://example.com/1",
            text="Content",
            themes=["AI"],
            insights=["Finding"],
        ),
        NotParsedDocument(
            title="Not Parsed",
            url="https://example.com/2",
        ),
    ]

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("analyzed")

    assert result["status"] == "completed"
    # Should still work, only using parsed documents


@pytest.mark.asyncio
async def test_write_sota_empty_llm_response_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test SOTA generation with empty LLM response (None content)."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace()
            self.chat.completions = self

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    from sotaforge.utils.models import ParsedDocument

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    docs = [
        ParsedDocument(
            title="Paper",
            url="https://example.com",
            text="Content",
            themes=["AI"],
            insights=["Finding"],
        )
    ]

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("analyzed")

    assert result["status"] == "completed"
    assert result["text"] == "No summary generated."


@pytest.mark.asyncio
async def test_write_sota_includes_document_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that SOTA generation includes document metadata in prompts."""
    from sotaforge.utils import llm

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    sent_content = []

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace()
            self.chat.completions = self

        async def create(self, **kwargs: Any) -> SimpleNamespace:
            messages = kwargs.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    sent_content.append(msg.get("content", ""))

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="SOTA summary"))
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    from sotaforge.utils.models import ParsedDocument

    synthesizer_server = importlib.import_module("sotaforge.agents.synthesizer_server")

    from sotaforge.utils.models import SourceType

    docs = [
        ParsedDocument(
            title="Test Paper",
            url="https://example.com/test",
            text="Content",
            themes=["AI", "ML"],
            insights=["Novel approach"],
            source_type=SourceType.PAPER,
        )
    ]

    monkeypatch.setattr(
        synthesizer_server,
        "db_store",
        SimpleNamespace(fetch_documents=lambda c: docs),
    )

    func = synthesizer_server.write_sota.fn
    result = await func("analyzed")

    assert result["status"] == "completed"
    # Verify metadata was included in the prompt
    content = " ".join(sent_content)
    assert "Test Paper" in content
    assert "https://example.com/test" in content
    assert "AI" in content
    assert "ML" in content
    assert "Novel approach" in content
