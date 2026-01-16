"""Tests for `filter_server`."""

from types import SimpleNamespace
from typing import Any

import pytest


async def _fake_run_filter(prompt: str, good_title: str) -> SimpleNamespace:
    if good_title in prompt:
        out = SimpleNamespace(
            criterion_1=5,
            criterion_2=4,
            criterion_3=4,
            criterion_4=5,
            criterion_5=4,
        )
    else:
        out = SimpleNamespace(
            criterion_1=1,
            criterion_2=1,
            criterion_3=1,
            criterion_4=1,
            criterion_5=1,
        )
    return SimpleNamespace(output=out)


class FakeAgent:
    """Fake agent for testing filter functionality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize fake agent."""
        self._good_title = kwargs.get("good_title", "Doc1")

    async def run(self, prompt: str) -> SimpleNamespace:
        """Run fake scoring."""
        return await _fake_run_filter(prompt, self._good_title)


@pytest.mark.asyncio
async def test_filter_results_keeps_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that documents are correctly kept or filtered based on scores."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "text": "Content about topic."},
        {"title": "Doc2", "text": "Other content."},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    def make_agent(*args: Any, **kwargs: Any) -> FakeAgent:
        return FakeAgent(*args, **{**kwargs, "good_title": "Doc1"})

    monkeypatch.setattr(filter_server, "Agent", make_agent)

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    assert res["count"] == 1
    titles = [d["title"] for d in res["results"]]
    assert "Doc1" in titles and "Doc2" not in titles


@pytest.mark.asyncio
async def test_filter_results_no_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test filtering when collection is empty."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: [])
    )

    func = filter_server.filter_results.fn
    res = await func("q", "col", ["a", "b", "c", "d", "e"])

    assert res["count"] == 0
    assert res["scored_documents"] == []
    assert res["filtered_results"] == []
    assert res["query"] == "q"
    assert res["collection"] == "col"


@pytest.mark.asyncio
async def test_filter_results_all_documents_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test when all documents score high and pass the filter."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "text": "Content"},
        {"title": "Doc2", "text": "Content"},
        {"title": "Doc3", "text": "Content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    class HighScoreAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=5,
                    criterion_2=5,
                    criterion_3=5,
                    criterion_4=5,
                    criterion_5=5,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: HighScoreAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    assert res["count"] == 3
    assert len(res["results"]) == 3
    assert len(res["scored_documents"]) == 3
    # All should have high mean scores
    for scored_doc in res["scored_documents"]:
        assert scored_doc["mean_score"] == 5.0
        assert scored_doc["keep"] is True


@pytest.mark.asyncio
async def test_filter_results_all_documents_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test when all documents score low and fail the filter."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "text": "Content"},
        {"title": "Doc2", "text": "Content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    class LowScoreAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=1,
                    criterion_2=1,
                    criterion_3=1,
                    criterion_4=1,
                    criterion_5=1,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: LowScoreAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    assert res["count"] == 0
    assert len(res["results"]) == 0
    assert len(res["scored_documents"]) == 2
    # All should have low mean scores
    for scored_doc in res["scored_documents"]:
        assert scored_doc["mean_score"] == 1.0
        assert scored_doc["keep"] is False


@pytest.mark.asyncio
async def test_filter_results_boundary_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test filtering with score exactly at threshold (mean = 2)."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "text": "Content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    class BoundaryScoreAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            # Mean of 2.0 exactly
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=2,
                    criterion_2=2,
                    criterion_3=2,
                    criterion_4=2,
                    criterion_5=2,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: BoundaryScoreAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    # Mean = 2 should NOT pass (keep requires > 2)
    assert res["count"] == 0
    assert res["scored_documents"][0]["mean_score"] == 2.0
    assert res["scored_documents"][0]["keep"] is False


@pytest.mark.asyncio
async def test_filter_results_mixed_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test filtering with documents having different scores."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "High", "text": "Content"},
        {"title": "Medium", "text": "Content"},
        {"title": "Low", "text": "Content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    class VariableScoreAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.call_count = 0

        async def run(self, prompt: str) -> SimpleNamespace:
            self.call_count += 1
            if "High" in prompt:
                scores = (4, 5, 4, 5, 4)  # mean = 4.4
            elif "Medium" in prompt:
                scores = (3, 3, 2, 3, 2)  # mean = 2.6
            else:
                scores = (1, 2, 1, 2, 1)  # mean = 1.4
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=scores[0],
                    criterion_2=scores[1],
                    criterion_3=scores[2],
                    criterion_4=scores[3],
                    criterion_5=scores[4],
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: VariableScoreAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    # Only High and Medium should pass (mean > 2)
    assert res["count"] == 2
    result_titles = [d["title"] for d in res["results"]]
    assert "High" in result_titles
    assert "Medium" in result_titles
    assert "Low" not in result_titles


@pytest.mark.asyncio
async def test_filter_results_handles_scoring_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that scoring exceptions are handled gracefully."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "text": "Content"},
        {"title": "Doc2", "text": "Content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    class ErrorAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.call_count = 0

        async def run(self, prompt: str) -> SimpleNamespace:
            self.call_count += 1
            if self.call_count == 1:
                raise Exception("Scoring failed")
            # Second call succeeds
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=4,
                    criterion_2=4,
                    criterion_3=4,
                    criterion_4=4,
                    criterion_5=4,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: ErrorAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    # Only one document should be scored/filtered (the second one)
    assert res["count"] == 1
    assert len(res["scored_documents"]) == 1


@pytest.mark.asyncio
async def test_filter_results_uses_snippet_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that filter uses snippet when available."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "snippet": "Important snippet content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    prompts_received = []

    class TrackingAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            prompts_received.append(prompt)
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=3,
                    criterion_2=3,
                    criterion_3=3,
                    criterion_4=3,
                    criterion_5=3,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: TrackingAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    await func("query", "collection", criteria)

    assert len(prompts_received) == 1
    assert "Important snippet content" in prompts_received[0]


@pytest.mark.asyncio
async def test_filter_results_uses_abstract_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that filter uses abstract when snippet is not available."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    docs = [
        {"title": "Doc1", "abstract": "Important abstract content"},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    prompts_received = []

    class TrackingAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            prompts_received.append(prompt)
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=3,
                    criterion_2=3,
                    criterion_3=3,
                    criterion_4=3,
                    criterion_5=3,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: TrackingAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    await func("query", "collection", criteria)

    assert len(prompts_received) == 1
    assert "Important abstract content" in prompts_received[0]


@pytest.mark.asyncio
async def test_filter_results_handles_parsed_document_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that filter works with ParsedDocument objects, not just dicts."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")
    from sotaforge.utils.models import ParsedDocument

    docs = [
        ParsedDocument(title="Doc1", text="Content"),
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    class SimpleAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=4,
                    criterion_2=4,
                    criterion_3=4,
                    criterion_4=4,
                    criterion_5=4,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: SimpleAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    res = await func("query", "collection", criteria)

    assert res["count"] == 1
    assert res["results"][0]["title"] == "Doc1"


@pytest.mark.asyncio
async def test_filter_results_truncates_long_excerpts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that long excerpts are truncated in prompts."""
    import importlib

    filter_server = importlib.import_module("sotaforge.agents.filter_server")

    long_text = "x" * 1000
    docs = [
        {"title": "Doc1", "text": long_text},
    ]

    monkeypatch.setattr(
        filter_server, "db_store", SimpleNamespace(fetch_documents=lambda c: docs)
    )

    prompts_received = []

    class TrackingAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def run(self, prompt: str) -> SimpleNamespace:
            prompts_received.append(prompt)
            return SimpleNamespace(
                output=SimpleNamespace(
                    criterion_1=3,
                    criterion_2=3,
                    criterion_3=3,
                    criterion_4=3,
                    criterion_5=3,
                )
            )

    monkeypatch.setattr(filter_server, "Agent", lambda *a, **k: TrackingAgent())

    criteria = ["relevance", "novelty", "impact", "clarity", "timeliness"]

    func = filter_server.filter_results.fn
    await func("query", "collection", criteria)

    assert len(prompts_received) == 1
    # Should be truncated to 600 characters
    assert long_text[:600] in prompts_received[0]
    assert len(prompts_received[0]) < len(long_text)
