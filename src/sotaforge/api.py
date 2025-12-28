"""FastAPI server for SOTAforge - REST API for frontend integration."""

import importlib
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)


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
    title="SOTAforge API",
    description="REST API for generating State-of-the-Art research summaries",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS - CRITICAL for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js default dev server
        "http://localhost:3001",
        "http://127.0.0.1:3000",
    ],
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
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Detailed health check."""
    return {
        "status": "healthy",
        "chroma_path": os.getenv("SOTAFORGE_CHROMA_PATH", "Not set"),
    }


@app.post("/api/sota", response_model=SOTAResponse)
async def generate_sota(request: SOTARequest) -> SOTAResponse:
    """Generate a State-of-the-Art research summary on a given topic.

    Args:
        request: SOTARequest containing the research topic

    Returns:
        SOTAResponse with generation status and results

    Raises:
        HTTPException: If generation fails

    """
    # Create a fresh Chroma DB directory for this request
    temp_chroma_dir = tempfile.mkdtemp(prefix="sotaforge_sota_")
    previous_chroma_path = os.environ.get("SOTAFORGE_CHROMA_PATH")

    try:
        topic = request.topic.strip()

        if not topic:
            raise HTTPException(status_code=400, detail="Topic cannot be empty")

        # Set the environment variable for this specific request
        os.environ["SOTAFORGE_CHROMA_PATH"] = temp_chroma_dir
        logger.info(f"Starting SOTA generation for: {topic}")
        logger.info(f"Using temporary Chroma DB: {temp_chroma_dir}")

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
            raise HTTPException(status_code=500, detail=msg)

        # Import (or reload) MCP agents agter
        # setting the env var so they bind to this temp DB
        module_names = [
            "sotaforge.agents.db_agent",
            "sotaforge.agents.filter_agent",
            "sotaforge.agents.parser_server",
            "sotaforge.agents.analyzer_server",
            "sotaforge.agents.synthesizer_server",
            "sotaforge.agents.orchestrator",
        ]
        for name in module_names:
            if name in sys.modules:
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)

        orchestrator = sys.modules["sotaforge.agents.orchestrator"]
        # Call the orchestrator (now bound to the per-request DB)
        result = await orchestrator.run_llm_sota(topic)

        logger.info(f"SOTA generation completed for: {topic}")

        return SOTAResponse(
            status="completed",
            result=result,
        )

    except Exception as e:
        logger.error(f"Error generating SOTA: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate SOTA: {str(e)}",
        )
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary Chroma DB: {temp_chroma_dir}")

        # Restore the previous environment variable
        if previous_chroma_path:
            os.environ["SOTAFORGE_CHROMA_PATH"] = previous_chroma_path
        elif "SOTAFORGE_CHROMA_PATH" in os.environ:
            del os.environ["SOTAFORGE_CHROMA_PATH"]


@app.get("/api/sota/status/{task_id}")
async def get_sota_status(task_id: str) -> Dict[str, str]:
    """Get the status of a SOTA generation task.

    Args:
        task_id: Unique identifier for the generation task

    Returns:
        Task status information

    """
    # TODO: Implement async task tracking with Celery or similar
    return {
        "task_id": task_id,
        "status": "not_implemented",
        "message": "Async task tracking coming soon",
    }


def main() -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "sotaforge.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
    )


if __name__ == "__main__":
    main()
