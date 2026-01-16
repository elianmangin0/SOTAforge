"""Tests for sotaforge.utils.mail."""

from types import SimpleNamespace
from typing import Any

import pytest

from sotaforge.utils import mail


def test_generate_pdf_starts_with_pdf() -> None:
    """Test that generated PDF starts with PDF magic bytes."""
    result = {"text": "Hello\n\nWorld", "status": "ok"}
    pdf = mail.generate_pdf("My Topic", result)
    assert isinstance(pdf, (bytes, bytearray))
    assert pdf[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_send_email_skips_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Test that send_email skips when API key is missing."""
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.delenv("SENDER_EMAIL", raising=False)
    caplog.set_level("WARNING")

    await mail.send_email("to@example.com", "Topic", {"text": "x", "status": "ok"})

    assert "RESEND_API_KEY not configured" in caplog.text


@pytest.mark.asyncio
async def test_send_email_calls_resend_when_api_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that send_email calls resend when API key is set."""
    monkeypatch.setenv("RESEND_API_KEY", "fake-key")
    monkeypatch.setenv("SENDER_EMAIL", "sender@example.com")

    sent = {}

    def fake_send(params: dict[str, Any]) -> dict[str, str]:
        sent["params"] = params
        return {"id": "fake-id"}

    fake_resend = SimpleNamespace(Emails=SimpleNamespace(send=fake_send))
    monkeypatch.setattr(mail, "resend", fake_resend)

    result = {"text": "abc", "status": "ready"}
    await mail.send_email("recipient@example.com", "Topic", result)

    assert "params" in sent
    params = sent["params"]
    assert params["to"] == ["recipient@example.com"]
    assert "attachments" in params
    filenames = [a["filename"] for a in params["attachments"]]
    assert any(f.endswith(".pdf") for f in filenames)
    assert any(f.endswith(".md") for f in filenames)
