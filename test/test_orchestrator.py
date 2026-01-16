"""Tests for `orchestrator`."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_emit_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that emit_progress correctly pushes to the queue."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    # Create a test queue
    test_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    orchestrator.progress_queue = test_queue  # type: ignore[attr-defined]

    await orchestrator.emit_progress("searching", "Test message", "search")

    # Check that message was added to queue
    assert test_queue.qsize() == 1
    msg = await test_queue.get()
    assert msg["status"] == "searching"
    assert msg["message"] == "Test message"
    assert msg["step"] == "search"
    assert "timestamp" in msg

    # Clean up
    orchestrator.progress_queue = None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_emit_progress_no_queue() -> None:
    """Test that emit_progress handles missing queue gracefully."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    # Ensure queue is None
    orchestrator.progress_queue = None  # type: ignore[attr-defined]

    # Should not raise an error
    await orchestrator.emit_progress("test", "message", "step")


@pytest.mark.asyncio
async def test_normalize_tool_result_with_sdk_result() -> None:
    """Test normalizing SDK-style tool results."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    # SDK-style result with content
    sdk_result = SimpleNamespace(content=[SimpleNamespace(text='{"key": "value"}')])

    normalized = orchestrator._normalize_tool_result(sdk_result)
    assert normalized == {"key": "value"}


@pytest.mark.asyncio
async def test_normalize_tool_result_with_dict() -> None:
    """Test normalizing dict results."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    result = {"key": "value"}
    normalized = orchestrator._normalize_tool_result(result)
    assert normalized == {"key": "value"}


@pytest.mark.asyncio
async def test_normalize_tool_result_with_json_string() -> None:
    """Test normalizing JSON string results."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    result = '{"key": "value"}'
    normalized = orchestrator._normalize_tool_result(result)
    assert normalized == {"key": "value"}


@pytest.mark.asyncio
async def test_normalize_tool_result_with_plain_string() -> None:
    """Test normalizing plain string results."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    result = "plain text"
    normalized = orchestrator._normalize_tool_result(result)
    assert normalized == "plain text"


@pytest.mark.asyncio
async def test_trim_message_history_under_limit() -> None:
    """Test that message history under limit is not trimmed."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": f"Message {i}"} for i in range(5)]

    trimmed = orchestrator._trim_message_history(messages, max_messages=10)

    assert len(trimmed) == 5
    assert trimmed == messages


@pytest.mark.asyncio
async def test_trim_message_history_over_limit() -> None:
    """Test that message history over limit is trimmed."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": f"Message {i}"} for i in range(20)]

    trimmed = orchestrator._trim_message_history(messages, max_messages=10)

    assert len(trimmed) == 10
    assert trimmed == messages[-10:]


@pytest.mark.asyncio
async def test_trim_message_history_preserves_tool_pairs() -> None:
    """Test that trimming preserves tool_call/tool pairs."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "Tool response"},
        {"role": "user", "content": "Message 2"},
    ]

    trimmed = orchestrator._trim_message_history(messages, max_messages=2)

    # Should include the tool call before the tool response
    assert len(trimmed) >= 2
    assert any(msg.get("role") == "tool" for msg in trimmed)


@pytest.mark.asyncio
async def test_get_last_messages() -> None:
    """Test getting last N messages."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": f"Message {i}"} for i in range(10)]

    last_5 = orchestrator._get_last_messages(messages, n=5)

    assert len(last_5) == 5
    assert last_5 == messages[-5:]


@pytest.mark.asyncio
async def test_get_last_messages_less_than_n() -> None:
    """Test getting last messages when fewer than N exist."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": f"Message {i}"} for i in range(3)]

    last_5 = orchestrator._get_last_messages(messages, n=5)

    assert len(last_5) == 3
    assert last_5 == messages


@pytest.mark.asyncio
async def test_extract_synthesized_sota_text_found() -> None:
    """Test extracting SOTA text from messages."""
    import importlib
    import json

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [
        {"role": "user", "content": "Generate SOTA"},
        {
            "role": "tool",
            "tool_call_id": "123",
            "content": json.dumps(
                {
                    "tool_name": "synthesizer_write_sota",
                    "result": {
                        "status": "completed",
                        "text": "This is the SOTA summary",
                    },
                }
            ),
        },
    ]

    sota_text = orchestrator._extract_synthesized_sota_text(messages)

    assert sota_text == "This is the SOTA summary"


@pytest.mark.asyncio
async def test_extract_synthesized_sota_text_not_found() -> None:
    """Test extracting SOTA text when not present."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [
        {"role": "user", "content": "Some message"},
        {"role": "assistant", "content": "Some response"},
    ]

    sota_text = orchestrator._extract_synthesized_sota_text(messages)

    assert sota_text == ""


@pytest.mark.asyncio
async def test_extract_synthesized_sota_text_unprefixed_tool() -> None:
    """Test extracting SOTA text with unprefixed tool name."""
    import importlib
    import json

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [
        {
            "role": "tool",
            "tool_call_id": "123",
            "content": json.dumps(
                {"tool_name": "write_sota", "result": {"text": "SOTA content"}}
            ),
        }
    ]

    sota_text = orchestrator._extract_synthesized_sota_text(messages)

    assert sota_text == "SOTA content"


@pytest.mark.asyncio
async def test_extract_synthesized_sota_text_alternative_key() -> None:
    """Test extracting SOTA text with 'sota' key instead of 'text'."""
    import importlib
    import json

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [
        {
            "role": "tool",
            "tool_call_id": "123",
            "content": json.dumps(
                {
                    "tool_name": "synthesizer_write_sota",
                    "result": {"sota": "SOTA via sota key"},
                }
            ),
        }
    ]

    sota_text = orchestrator._extract_synthesized_sota_text(messages)

    assert sota_text == "SOTA via sota key"


@pytest.mark.asyncio
async def test_execute_tool_calls_no_calls() -> None:
    """Test execute_tool_calls with no tool calls."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Response without tool calls"},
    ]

    result = await orchestrator._execute_tool_calls(messages)

    assert len(result) == 2
    assert result == messages


@pytest.mark.asyncio
async def test_validate_step_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that validation step returns approved response."""
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
                            content="APPROVE: This step looks good.",
                            tool_calls=None,
                            model_dump=lambda exclude_unset=True: {
                                "role": "assistant",
                                "content": "APPROVE: This step looks good.",
                            },
                        )
                    )
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": "Initial message"}]
    initial_length = len(messages)
    approved, updated_messages = await orchestrator.validate_step(
        messages, "Validate this step", []
    )

    assert approved is True
    assert len(updated_messages) > initial_length
    assert updated_messages is messages  # Same list reference


@pytest.mark.asyncio
async def test_validate_step_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that validation step returns rejected response."""
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
                            content="REDO: This step needs improvement.",
                            tool_calls=None,
                            model_dump=lambda exclude_unset=True: {
                                "role": "assistant",
                                "content": "REDO: This step needs improvement.",
                            },
                        )
                    )
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": "Initial message"}]
    initial_length = len(messages)
    approved, updated_messages = await orchestrator.validate_step(
        messages, "Validate this step", []
    )

    assert approved is False
    assert len(updated_messages) > initial_length
    assert updated_messages is messages  # Same list reference


@pytest.mark.asyncio
async def test_validate_step_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that validation is case-insensitive."""
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
                            content="approve: looks good",
                            tool_calls=None,
                            model_dump=lambda exclude_unset=True: {
                                "role": "assistant",
                                "content": "approve: looks good",
                            },
                        )
                    )
                ]
            )

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    messages = [{"role": "user", "content": "Test"}]
    approved, _ = await orchestrator.validate_step(messages, "Validate", [])

    assert approved is True


@pytest.mark.asyncio
async def test_emit_tool_progress_search_web(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test progress emission for search_web tool."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    test_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    orchestrator.progress_queue = test_queue  # type: ignore[attr-defined]

    await orchestrator._emit_tool_progress("search_web", {"query": "AI research"})

    assert test_queue.qsize() == 1
    msg = await test_queue.get()
    assert "AI research" in msg["message"]

    orchestrator.progress_queue = None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_emit_tool_progress_search_papers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test progress emission for search_papers tool."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    test_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    orchestrator.progress_queue = test_queue  # type: ignore[attr-defined]

    await orchestrator._emit_tool_progress(
        "search_papers", {"query": "machine learning"}
    )

    assert test_queue.qsize() == 1
    msg = await test_queue.get()
    assert "machine learning" in msg["message"]

    orchestrator.progress_queue = None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_emit_tool_progress_no_queue() -> None:
    """Test that emit_tool_progress handles missing queue gracefully."""
    import importlib

    orchestrator = importlib.import_module("sotaforge.agents.orchestrator")

    orchestrator.progress_queue = None  # type: ignore[attr-defined]

    # Should not raise an error
    await orchestrator._emit_tool_progress("search_web", {"query": "test"})
