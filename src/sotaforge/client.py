"""Client to interact with the SOTAforge orchestrator."""

import asyncio
import atexit
import os
import shutil
import tempfile
from typing import Any, Dict


def _configure_temp_chroma_path() -> str:
    """Create an empty temp Chroma dir per client run and clean it up on exit."""
    temp_dir = tempfile.mkdtemp(prefix="sotaforge_chroma_")
    os.environ["SOTAFORGE_CHROMA_PATH"] = temp_dir
    atexit.register(shutil.rmtree, temp_dir, True)
    return temp_dir


async def call_produce_sota(topic: str) -> Dict[str, Any]:
    """Produce a SOTA on the given topic by calling the orchestrator directly.

    Args:
        topic: The research topic (must be non-empty)

    Raises:
        ValueError: If topic is empty or invalid

    """
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")

    topic = topic.strip()

    # Import after setting the temp DB path so orchestrator/DB use it
    from sotaforge.agents.orchestrator import run_llm_sota

    result = await run_llm_sota(topic)
    return {"status": "done", "result": result}


async def _run_client() -> None:
    """Run the client to generate a SOTA with a fresh temp DB."""
    temp_path = _configure_temp_chroma_path()

    topic = input("Enter topic for SOTA generation: ").strip()
    if not topic:
        topic = "pdf parsing technologies"
        print(f"Using default topic: {topic}")

    print(f"Using temporary Chroma path: {temp_path}")
    print(f"\n🚀 Generating SOTA on: {topic}\n")
    await call_produce_sota(topic)


def main() -> None:
    """Entry point for the client."""
    asyncio.run(_run_client())


if __name__ == "__main__":
    main()
