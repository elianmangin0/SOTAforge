"""Tests for `db_server`."""

from typing import Any

import pytest

from sotaforge.utils.errors import DatabaseError


@pytest.mark.asyncio
async def test_store_records_with_parsed_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing parsed documents with text field."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["collection"] = collection
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(
        db_server.store,
        "upsert_documents",
        fake_upsert,
    )

    items = [
        {"title": "Doc1", "text": "Content 1", "url": "http://example.com/1"},
        {"title": "Doc2", "text": "Content 2", "url": "http://example.com/2"},
    ]

    func = db_server.store_records.fn
    res = await func("test_collection", items)

    assert res["collection"] == "test_collection"
    assert res["count"] == 2
    assert res["ids"] == ["id_0", "id_1"]
    assert stored_data["collection"] == "test_collection"
    assert len(stored_data["docs"]) == 2


@pytest.mark.asyncio
async def test_store_records_with_not_parsed_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing not parsed documents without text field."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["collection"] = collection
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(
        db_server.store,
        "upsert_documents",
        fake_upsert,
    )

    items = [
        {"title": "Doc1", "url": "http://example.com/1", "snippet": "Snippet 1"},
        {"title": "Doc2", "url": "http://example.com/2", "abstract": "Abstract 2"},
    ]

    func = db_server.store_records.fn
    res = await func("test_collection", items)

    assert res["collection"] == "test_collection"
    assert res["count"] == 2
    assert res["ids"] == ["id_0", "id_1"]


@pytest.mark.asyncio
async def test_store_records_mixed_documents_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that mixing parsed and not parsed documents raises an error."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    items = [
        {"title": "Doc1", "text": "Content 1"},  # ParsedDocument
        {"title": "Doc2", "url": "http://example.com/2"},  # NotParsedDocument
    ]

    func = db_server.store_records.fn

    with pytest.raises(DatabaseError, match="must contain either"):
        await func("test_collection", items)


@pytest.mark.asyncio
async def test_store_records_invalid_items_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that non-dict items raise an error."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    items = ["not a dict", {"title": "Doc1"}]

    func = db_server.store_records.fn

    with pytest.raises(DatabaseError, match="All items must be dictionaries"):
        await func("test_collection", items)


@pytest.mark.asyncio
async def test_fetch_documents_returns_all_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fetching all documents from a collection."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")
    from sotaforge.utils.models import ParsedDocument

    docs = [
        ParsedDocument(title="Doc1", text="Content 1"),
        ParsedDocument(title="Doc2", text="Content 2"),
    ]

    monkeypatch.setattr(
        db_server.store,
        "fetch_documents",
        lambda collection, limit=None: docs,
    )

    func = db_server.fetch_documents.fn
    res = await func("test_collection")

    assert res["collection"] == "test_collection"
    assert res["count"] == 2
    assert len(res["documents"]) == 2
    assert res["documents"][0]["title"] == "Doc1"
    assert res["documents"][1]["title"] == "Doc2"


@pytest.mark.asyncio
async def test_fetch_documents_with_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching documents with a limit."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")
    from sotaforge.utils.models import ParsedDocument

    docs = [ParsedDocument(title="Doc1", text="Content 1")]

    fetch_calls: dict[str, Any] = {}

    def fake_fetch(collection: str, limit: int | None = None) -> list[Any]:
        fetch_calls["collection"] = collection
        fetch_calls["limit"] = limit
        return docs

    monkeypatch.setattr(db_server.store, "fetch_documents", fake_fetch)

    func = db_server.fetch_documents.fn
    res = await func("test_collection", limit=5)

    assert res["count"] == 1
    assert fetch_calls["limit"] == 5


@pytest.mark.asyncio
async def test_fetch_documents_empty_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fetching from an empty collection."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    monkeypatch.setattr(
        db_server.store,
        "fetch_documents",
        lambda collection, limit=None: [],
    )

    func = db_server.fetch_documents.fn
    res = await func("empty_collection")

    assert res["collection"] == "empty_collection"
    assert res["count"] == 0
    assert res["documents"] == []


@pytest.mark.asyncio
async def test_store_tool_results_no_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test store_tool_results when no messages are provided."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["tool_id_1"], messages=None)

    assert "error" in res
    assert res["count"] == 0


@pytest.mark.asyncio
async def test_store_tool_results_with_results_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing tool results with 'results' field in response."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["collection"] = collection
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(db_server.store, "upsert_documents", fake_upsert)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"results": [{"title": "Doc1", "text": "Content"}]}',
        }
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_123"], messages=messages)

    assert res["collection"] == "test_collection"
    assert res["count"] == 1
    assert len(stored_data["docs"]) == 1


@pytest.mark.asyncio
async def test_store_tool_results_with_result_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing tool results with orchestrator 'result' wrapper."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(db_server.store, "upsert_documents", fake_upsert)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": '{"result": {"results": [{"title": "Doc2", "text": "Text"}]}}',
        }
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_456"], messages=messages)

    assert res["count"] == 1
    assert len(stored_data["docs"]) == 1


@pytest.mark.asyncio
async def test_store_tool_results_with_list_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing tool results when content is a direct list."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(db_server.store, "upsert_documents", fake_upsert)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_789",
            "content": (
                '[{"title": "Doc3", "text": "Content 3"}, '
                '{"title": "Doc4", "text": "Content 4"}]'
            ),
        }
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_789"], messages=messages)

    assert res["count"] == 2
    assert len(stored_data["docs"]) == 2


@pytest.mark.asyncio
async def test_store_tool_results_with_single_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing tool results when content is a single dict."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(db_server.store, "upsert_documents", fake_upsert)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_single",
            "content": '{"title": "Single Doc", "text": "Single content"}',
        }
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_single"], messages=messages)

    assert res["count"] == 1
    assert len(stored_data["docs"]) == 1


@pytest.mark.asyncio
async def test_store_tool_results_no_matching_tool_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test store_tool_results when tool_call_id doesn't match any message."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_different",
            "content": '{"title": "Doc", "text": "Content"}',
        }
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_missing"], messages=messages)

    assert res["count"] == 0
    assert "message" in res


@pytest.mark.asyncio
async def test_store_tool_results_multiple_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test storing results from multiple tool calls."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(db_server.store, "upsert_documents", fake_upsert)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"title": "Doc1", "text": "Content 1"}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"title": "Doc2", "text": "Content 2"}',
        },
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_1", "call_2"], messages=messages)

    assert res["count"] == 2
    assert len(stored_data["docs"]) == 2


@pytest.mark.asyncio
async def test_store_tool_results_filters_non_dict_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that non-dict items in results are filtered out."""
    import importlib

    db_server = importlib.import_module("sotaforge.agents.db_server")

    stored_data: dict[str, Any] = {}

    def fake_upsert(collection: str, docs: list[Any]) -> list[str]:
        stored_data["docs"] = docs
        return [f"id_{i}" for i in range(len(docs))]

    monkeypatch.setattr(db_server.store, "upsert_documents", fake_upsert)

    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_mixed",
            "content": (
                '{"results": [{"title": "Doc1", "text": "Content"}, "not a dict", 123]}'
            ),
        }
    ]

    func = db_server.store_tool_results.fn
    res = await func("test_collection", ["call_mixed"], messages=messages)

    assert res["count"] == 1
    assert len(stored_data["docs"]) == 1
