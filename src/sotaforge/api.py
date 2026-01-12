"""FastAPI server for SOTAforge - REST API for frontend integration."""

import asyncio
import importlib
import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sotaforge.utils.constants import (
    ALLOWED_ORIGINS,
    API_DESCRIPTION,
    API_HOST,
    API_PORT,
    API_TITLE,
)
from sotaforge.utils.logger import get_logger

# Load environment variables from .env.secrets file
load_dotenv(".env.secrets")

logger = get_logger(__name__)

try:
    from importlib.metadata import version

    __version__ = version("sotaforge")
except Exception:
    __version__ = "0.1.0"

# In-memory task storage with progress tracking
tasks: Dict[str, Dict[str, Any]] = {}
progress_queues: Dict[str, asyncio.Queue[Any]] = {}


class SOTARequest(BaseModel):
    """Request model for SOTA generation."""

    topic: str = Field(
        ..., min_length=1, description="Research topic for SOTA generation"
    )


class SOTAResponse(BaseModel):
    """Response model for SOTA generation."""

    status: str
    result: Dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    logger.info("SOTAforge API starting up")
    yield
    logger.info("SOTAforge API shutting down")


# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS - CRITICAL for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "message": "SOTAforge API is running",
        "status": "healthy",
        "version": __version__,
    }


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check() -> Dict[str, str]:
    """Detailed health check."""
    return {
        "status": "healthy",
    }


@app.post("/api/sota")
async def generate_sota(
    request: SOTARequest, background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """Start a SOTA generation task and return task_id for streaming.

    Args:
        request: SOTARequest containing the research topic
        background_tasks: FastAPI background tasks

    Returns:
        Dictionary with task_id

    """
    task_id = str(uuid.uuid4())
    progress_queues[task_id] = asyncio.Queue()
    tasks[task_id] = {
        "status": "pending",
        "topic": request.topic,
        "created_at": datetime.now().isoformat(),
    }

    # Run the generation in the background
    background_tasks.add_task(run_sota_generation, task_id, request.topic)

    return {"task_id": task_id}


async def run_sota_generation(task_id: str, topic: str) -> None:
    """Run SOTA generation and emit progress updates.

    Args:
        task_id: Unique task identifier
        topic: Research topic

    """
    queue = progress_queues[task_id]
    temp_chroma_dir = tempfile.mkdtemp(prefix="sotaforge_sota_")
    previous_chroma_path = os.environ.get("SOTAFORGE_CHROMA_PATH")

    try:
        await queue.put(
            {
                "status": "initializing",
                "message": "Initializing SOTA generation system...",
                "step": "initializing",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Set the environment variable for this specific request
        os.environ["SOTAFORGE_CHROMA_PATH"] = temp_chroma_dir
        logger.info(f"Starting SOTA generation for: {topic}")
        logger.info(f"Using temporary Chroma DB: {temp_chroma_dir}")

        await queue.put(
            {
                "status": "initializing",
                "message": "Setting up isolated database environment...",
                "step": "initializing",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Preflight: ensure required API keys are present
        missing_keys = []
        for key in ("OPENAI_API_KEY", "SERPER_API_KEY"):
            if not os.getenv(key):
                missing_keys.append(key)
        if missing_keys:
            msg = (
                "Missing required environment variables: "
                + ", ".join(missing_keys)
                + ". Set them before calling the API."
            )
            logger.error(msg)
            await queue.put({"status": "failed", "message": msg})
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = msg
            return

        await queue.put(
            {
                "status": "loading",
                "message": "Verifying API credentials...",
                "step": "loading",
                "timestamp": datetime.now().isoformat(),
            }
        )

        await queue.put(
            {
                "status": "loading",
                "message": "Loading MCP agent modules...",
                "step": "loading",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Import (or reload) MCP agents after setting the env var
        module_names = [
            "sotaforge.agents.db_agent",
            "sotaforge.agents.filter_agent",
            "sotaforge.agents.parser_server",
            "sotaforge.agents.analyzer_server",
            "sotaforge.agents.synthesizer_server",
            "sotaforge.agents.orchestrator",
        ]
        for name in module_names:
            agent_name = name.split(".")[-1].replace("_", " ").title()
            await queue.put(
                {
                    "status": "loading",
                    "message": f"Loading {agent_name}...",
                    "step": "loading",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            if name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)

        await queue.put(
            {
                "status": "loading",
                "message": "Configuring orchestrator and injecting dependencies...",
                "step": "loading",
                "timestamp": datetime.now().isoformat(),
            }
        )

        orchestrator = sys.modules["sotaforge.agents.orchestrator"]
        # Inject the progress queue into orchestrator
        orchestrator.progress_queue = queue  # type: ignore[attr-defined]

        await queue.put(
            {
                "status": "running",
                "message": "All systems ready. Starting SOTA generation pipeline...",
                "step": "running",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Call the orchestrator (now bound to the per-request DB)
        result = await orchestrator.run_llm_sota(topic)

        logger.info(f"SOTA generation completed for: {topic}")

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        await queue.put(
            {
                "status": "completed",
                "message": "SOTA generation completed!",
                "result": result,
            }
        )

    except Exception as e:
        logger.error(f"Error generating SOTA: {str(e)}", exc_info=True)
        error_msg = f"Failed to generate SOTA: {str(e)}"
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = error_msg
        await queue.put({"status": "failed", "message": error_msg})
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary Chroma DB: {temp_chroma_dir}")

        # Restore the previous environment variable
        if previous_chroma_path:
            os.environ["SOTAFORGE_CHROMA_PATH"] = previous_chroma_path
        elif "SOTAFORGE_CHROMA_PATH" in os.environ:
            del os.environ["SOTAFORGE_CHROMA_PATH"]

        # Signal end of stream
        await queue.put(None)


@app.get("/api/sota/status/{task_id}")
async def get_sota_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a SOTA generation task.

    Args:
        task_id: Unique identifier for the generation task

    Returns:
        Task status information

    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks[task_id]


@app.get("/api/sota/stream/{task_id}")
async def stream_sota_progress(task_id: str) -> StreamingResponse:
    """Stream SOTA generation progress via Server-Sent Events.

    Args:
        task_id: Unique identifier for the generation task

    Returns:
        StreamingResponse with SSE events

    """
    if task_id not in progress_queues:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from the progress queue."""
        queue = progress_queues[task_id]

        try:
            while True:
                # Wait for next progress update
                progress = await queue.get()

                # None signals end of stream
                if progress is None:
                    break

                # Send as SSE event
                import json

                yield f"data: {json.dumps(progress)}\n\n"

        finally:
            # Clean up queue when done
            if task_id in progress_queues:
                del progress_queues[task_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "sotaforge.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # Auto-reload on code changes (dev only)
    )


if __name__ == "__main__":
    main()
