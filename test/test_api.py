"""Tests for the FastAPI REST API."""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client for the FastAPI app."""
    # Set required environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")

    # Import after setting env vars
    from sotaforge.api import app

    return TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    """Test the root endpoint returns basic info."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "message" in data
    assert "version" in data


def test_health_check_get(client: TestClient) -> None:
    """Test the health check endpoint with GET."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_check_head(client: TestClient) -> None:
    """Test the health check endpoint with HEAD."""
    response = client.head("/health")

    assert response.status_code == 200


def test_generate_sota_valid_request(client: TestClient) -> None:
    """Test generating SOTA with valid request."""
    response = client.post(
        "/api/sota", json={"topic": "AI research", "email": "test@example.com"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert len(data["task_id"]) > 0


def test_generate_sota_without_email(client: TestClient) -> None:
    """Test generating SOTA without email."""
    response = client.post("/api/sota", json={"topic": "Machine learning"})

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data


def test_generate_sota_empty_topic(client: TestClient) -> None:
    """Test that empty topic is rejected."""
    response = client.post("/api/sota", json={"topic": ""})

    assert response.status_code == 422  # Validation error


def test_generate_sota_missing_topic(client: TestClient) -> None:
    """Test that missing topic is rejected."""
    response = client.post("/api/sota", json={})

    assert response.status_code == 422  # Validation error


def test_get_sota_status_existing_task(client: TestClient) -> None:
    """Test getting status of existing task."""
    # First create a task
    create_response = client.post("/api/sota", json={"topic": "Deep learning"})
    task_id = create_response.json()["task_id"]

    # Then get its status
    response = client.get(f"/api/sota/status/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["topic"] == "Deep learning"
    assert "created_at" in data


def test_get_sota_status_nonexistent_task(client: TestClient) -> None:
    """Test getting status of nonexistent task."""
    response = client.get("/api/sota/status/nonexistent-task-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_cancel_sota_existing_task(client: TestClient) -> None:
    """Test cancelling an existing task."""
    # First create a task
    create_response = client.post("/api/sota", json={"topic": "Neural networks"})
    task_id = create_response.json()["task_id"]

    # Then cancel it
    response = client.delete(f"/api/sota/{task_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"

    # Verify task status was updated
    status_response = client.get(f"/api/sota/status/{task_id}")
    assert status_response.json()["status"] == "cancelled"


def test_cancel_sota_nonexistent_task(client: TestClient) -> None:
    """Test cancelling a nonexistent task."""
    response = client.delete("/api/sota/nonexistent-task-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_stream_sota_progress_nonexistent_task(client: TestClient) -> None:
    """Test streaming progress for nonexistent task."""
    response = client.get("/api/sota/stream/nonexistent-task-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_run_sota_generation_missing_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that missing API keys are detected."""
    # Remove API keys
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    from sotaforge.api import progress_queues, run_sota_generation, tasks

    task_id = "test-task-1"
    queue: asyncio.Queue[Any] = asyncio.Queue()
    progress_queues[task_id] = queue
    tasks[task_id] = {"status": "pending"}

    await run_sota_generation(task_id, "Test topic", "")

    # Check that task failed
    assert tasks[task_id]["status"] == "failed"
    assert "Missing required environment variables" in tasks[task_id]["error"]

    # Verify error was sent to queue
    messages = []
    while not queue.empty():
        msg = await queue.get()
        if msg is not None:
            messages.append(msg)

    error_messages = [m for m in messages if m.get("status") == "failed"]
    assert len(error_messages) > 0
    assert "Missing required environment variables" in error_messages[0]["message"]


@pytest.mark.asyncio
async def test_run_sota_generation_cancelled_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test task cancellation before it starts."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    from sotaforge.api import (
        cancelled_tasks,
        progress_queues,
        run_sota_generation,
        tasks,
    )

    task_id = "test-task-2"
    queue: asyncio.Queue[Any] = asyncio.Queue()
    progress_queues[task_id] = queue
    tasks[task_id] = {"status": "pending"}
    cancelled_tasks.add(task_id)  # Cancel before starting

    await run_sota_generation(task_id, "Test topic", "")

    # Check that task was cancelled
    assert tasks[task_id]["status"] == "cancelled"

    # Verify cancellation was sent to queue
    messages = []
    while not queue.empty():
        msg = await queue.get()
        if msg is not None:
            messages.append(msg)

    cancelled_messages = [m for m in messages if m.get("status") == "cancelled"]
    assert len(cancelled_messages) > 0


@pytest.mark.asyncio
async def test_run_sota_generation_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful SOTA generation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    # Mock the orchestrator's run_llm_sota function
    mock_result = {
        "topic": "AI research",
        "status": "completed",
        "text": "This is the generated SOTA summary",
    }

    async def mock_run_llm_sota(topic: str) -> Dict[str, Any]:
        return mock_result

    # Create a mock orchestrator module
    import importlib
    import sys
    from types import ModuleType

    mock_orchestrator = ModuleType("sotaforge.agents.orchestrator")
    mock_orchestrator.run_llm_sota = mock_run_llm_sota  # type: ignore[attr-defined]
    mock_orchestrator.progress_queue = None  # type: ignore[attr-defined]

    # Mock other agent modules to avoid loading them
    mock_modules = {
        "sotaforge.agents.orchestrator": mock_orchestrator,
        "sotaforge.agents.db_server": ModuleType("sotaforge.agents.db_server"),
        "sotaforge.agents.filter_server": ModuleType("sotaforge.agents.filter_server"),
        "sotaforge.agents.parser_server": ModuleType("sotaforge.agents.parser_server"),
        "sotaforge.agents.analyzer_server": ModuleType(
            "sotaforge.agents.analyzer_server"
        ),
        "sotaforge.agents.synthesizer_server": ModuleType(
            "sotaforge.agents.synthesizer_server"
        ),
    }

    # Mock importlib.reload and importlib.import_module to return our mocks
    original_import = importlib.import_module

    def mock_reload(module: Any) -> Any:
        return module

    def mock_import_module(name: str) -> Any:
        return mock_modules.get(name, original_import(name))

    with patch.dict(sys.modules, mock_modules):
        with patch("importlib.reload", side_effect=mock_reload):
            with patch("importlib.import_module", side_effect=mock_import_module):
                from sotaforge.api import progress_queues, run_sota_generation, tasks

                task_id = "test-task-3"
                queue: asyncio.Queue[Any] = asyncio.Queue()
                progress_queues[task_id] = queue
                tasks[task_id] = {"status": "pending"}

                await run_sota_generation(task_id, "AI research", "")

                # Check that task completed
                assert tasks[task_id]["status"] == "completed"
                assert tasks[task_id]["result"] == mock_result

                # Verify completion was sent to queue
                messages = []
                while not queue.empty():
                    msg = await queue.get()
                    if msg is not None:
                        messages.append(msg)

                completed_messages = [
                    m for m in messages if m.get("status") == "completed"
                ]
                assert len(completed_messages) > 0


@pytest.mark.asyncio
async def test_run_sota_generation_with_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test SOTA generation with email notification."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    # Mock the orchestrator and email
    mock_result = {"topic": "AI", "status": "completed", "text": "Summary"}

    async def mock_run_llm_sota(topic: str) -> Dict[str, Any]:
        return mock_result

    mock_send_email = AsyncMock()

    # Create a mock orchestrator module
    import importlib
    import sys
    from types import ModuleType

    mock_orchestrator = ModuleType("sotaforge.agents.orchestrator")
    mock_orchestrator.run_llm_sota = mock_run_llm_sota  # type: ignore[attr-defined]
    mock_orchestrator.progress_queue = None  # type: ignore[attr-defined]

    # Mock other agent modules to avoid loading them
    mock_modules = {
        "sotaforge.agents.orchestrator": mock_orchestrator,
        "sotaforge.agents.db_server": ModuleType("sotaforge.agents.db_server"),
        "sotaforge.agents.filter_server": ModuleType("sotaforge.agents.filter_server"),
        "sotaforge.agents.parser_server": ModuleType("sotaforge.agents.parser_server"),
        "sotaforge.agents.analyzer_server": ModuleType(
            "sotaforge.agents.analyzer_server"
        ),
        "sotaforge.agents.synthesizer_server": ModuleType(
            "sotaforge.agents.synthesizer_server"
        ),
    }

    # Mock importlib.reload and importlib.import_module to return our mocks
    original_import = importlib.import_module

    def mock_reload(module: Any) -> Any:
        return module

    def mock_import_module(name: str) -> Any:
        return mock_modules.get(name, original_import(name))

    # Patch both the orchestrator module and send_email
    with patch.dict(sys.modules, mock_modules):
        with patch("importlib.reload", side_effect=mock_reload):
            with patch("importlib.import_module", side_effect=mock_import_module):
                with patch("sotaforge.api.send_email", mock_send_email):
                    from sotaforge.api import (
                        progress_queues,
                        run_sota_generation,
                        tasks,
                    )

                    task_id = "test-task-4"
                    queue: asyncio.Queue[Any] = asyncio.Queue()
                    progress_queues[task_id] = queue
                    tasks[task_id] = {"status": "pending"}

                    await run_sota_generation(task_id, "AI", "test@example.com")

                    # Verify email was sent
                    mock_send_email.assert_called_once_with(
                        "test@example.com", "AI", mock_result
                    )


@pytest.mark.asyncio
async def test_run_sota_generation_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test error handling in SOTA generation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-key")

    # Mock orchestrator to raise error
    async def mock_run_llm_sota(topic: str) -> Dict[str, Any]:
        raise Exception("Orchestrator error")

    # Create a mock orchestrator module
    import importlib
    import sys
    from types import ModuleType

    mock_orchestrator = ModuleType("sotaforge.agents.orchestrator")
    mock_orchestrator.run_llm_sota = mock_run_llm_sota  # type: ignore[attr-defined]
    mock_orchestrator.progress_queue = None  # type: ignore[attr-defined]

    # Mock other agent modules to avoid loading them
    mock_modules = {
        "sotaforge.agents.orchestrator": mock_orchestrator,
        "sotaforge.agents.db_server": ModuleType("sotaforge.agents.db_server"),
        "sotaforge.agents.filter_server": ModuleType("sotaforge.agents.filter_server"),
        "sotaforge.agents.parser_server": ModuleType("sotaforge.agents.parser_server"),
        "sotaforge.agents.analyzer_server": ModuleType(
            "sotaforge.agents.analyzer_server"
        ),
        "sotaforge.agents.synthesizer_server": ModuleType(
            "sotaforge.agents.synthesizer_server"
        ),
    }

    # Mock importlib.reload and importlib.import_module to return our mocks
    original_import = importlib.import_module

    def mock_reload(module: Any) -> Any:
        return module

    def mock_import_module(name: str) -> Any:
        return mock_modules.get(name, original_import(name))

    # Patch the orchestrator module
    with patch.dict(sys.modules, mock_modules):
        with patch("importlib.reload", side_effect=mock_reload):
            with patch("importlib.import_module", side_effect=mock_import_module):
                from sotaforge.api import progress_queues, run_sota_generation, tasks

                task_id = "test-task-5"
                queue: asyncio.Queue[Any] = asyncio.Queue()
                progress_queues[task_id] = queue
                tasks[task_id] = {"status": "pending"}

                await run_sota_generation(task_id, "Test", "")

                # Check that task failed
                assert tasks[task_id]["status"] == "failed"
                assert "error" in tasks[task_id]
                assert "Orchestrator error" in tasks[task_id]["error"]


def test_cors_configuration(client: TestClient) -> None:
    """Test that CORS headers are properly configured."""
    response = client.options(
        "/api/sota",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    # CORS should be configured
    assert "access-control-allow-origin" in response.headers


def test_api_validates_request_schema(client: TestClient) -> None:
    """Test that API properly validates request schema."""
    # Invalid data type
    response = client.post(
        "/api/sota",
        json={"topic": 123},  # Should be string
    )

    assert response.status_code == 422


def test_multiple_concurrent_tasks(client: TestClient) -> None:
    """Test creating multiple tasks concurrently."""
    topics = ["AI", "ML", "Deep Learning", "NLP"]
    task_ids = []

    for topic in topics:
        response = client.post("/api/sota", json={"topic": topic})
        assert response.status_code == 200
        task_ids.append(response.json()["task_id"])

    # All task IDs should be unique
    assert len(task_ids) == len(set(task_ids))

    # All tasks should be retrievable
    for task_id in task_ids:
        response = client.get(f"/api/sota/status/{task_id}")
        assert response.status_code == 200
