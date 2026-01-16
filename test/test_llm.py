"""Tests for sotaforge.utils.llm."""

from typing import Any

import pytest

from sotaforge.utils import llm


def test_get_llm_raises_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_llm raises error when API key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm.reset_llm()
    with pytest.raises(ValueError):
        llm.get_llm()


def test_get_llm_returns_client_when_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_llm returns client when API key is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None) -> None:
            self.api_key = api_key

    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)
    llm.reset_llm()

    client = llm.get_llm()
    assert isinstance(client, DummyClient)
    assert client.api_key == "test-key"

    # Ensure cached instance is returned
    client2 = llm.get_llm()
    assert client is client2


def test_get_pydantic_model_uses_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_pydantic_model uses the configured provider and model."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class DummyClient:
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            self.api_key = api_key

    # Patch the AsyncOpenAI class so the cached `get_llm` creates a DummyClient
    monkeypatch.setattr(llm, "AsyncOpenAI", DummyClient)

    captured = {}

    def fake_provider(openai_client: Any) -> str:
        captured["client"] = openai_client
        return "provider-object"

    monkeypatch.setattr(
        llm, "OpenAIProvider", lambda openai_client: fake_provider(openai_client)
    )

    class FakeModel:
        def __init__(self, model_name: str, provider: Any = None) -> None:
            self.model_name = model_name
            self.provider = provider

    monkeypatch.setattr(llm, "OpenAIChatModel", FakeModel)

    llm.reset_llm()
    model = llm.get_pydantic_model("my-model")

    assert isinstance(model, FakeModel)
    assert model.model_name == "my-model"
    assert model.provider == "provider-object"
    assert "client" in captured
