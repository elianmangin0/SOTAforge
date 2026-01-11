"""Synthesizer server for generating state-of-the-art summaries."""

from typing import Dict, List, Union

from fastmcp import FastMCP
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from sotaforge.utils.constants import (
    MODEL,
    SYNTHESIZER_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from sotaforge.utils.dataclasses import ParsedDocument
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)
server = FastMCP("synthesizer")
llm = AsyncOpenAI()
db_store = ChromaStore()


@server.tool(
    name="write_sota",
    description="Writes a structured state-of-the-art based on analyzed documents",
)
async def write_sota(collection: str) -> Dict[str, Union[str, str]]:
    """Generate a structured state-of-the-art summary from analyzed documents."""
    logger.info(f"Synthesizing report from collection: {collection}")

    # Fetch all analyzed documents from the collection
    documents = db_store.fetch_documents(collection)
    logger.info(f"Retrieved {len(documents)} documents from '{collection}'")

    if not documents:
        logger.warning(f"No documents found in collection '{collection}'")
        return {
            "status": "error",
            "text": "No documents found in collection",
        }

    logger.info(f"Writing SOTA from {len(documents)} analyzed documents")

    # Filter only ParsedDocument instances (not NotParsedDocument)
    analyzed_docs = [doc for doc in documents if isinstance(doc, ParsedDocument)]

    # Format documents with sources
    docs_content = "\n\n".join(
        [
            (
                f"Source: {doc.title}\nURL: {doc.url}\n"
                f"Type: {doc.source_type}\n"
                f"Themes: {', '.join(doc.themes) if doc.themes else 'N/A'}\n"
                f"Insights: {', '.join(doc.insights) if doc.insights else 'N/A'}\n"
                f"Content: {
                    doc.text
                    if len(doc.text) < 1500
                    else (doc.snippet or doc.abstract or 'N/A')
                }"
            )
            for doc in analyzed_docs
        ]
    )

    logger.debug(f"Formatted documents for synthesis: {docs_content}")
    messages: List[
        ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
    ] = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"{SYNTHESIZER_PROMPT} {docs_content}"),
        },
    ]

    response = await llm.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5,
    )

    content = response.choices[0].message.content or ""
    text = content.strip() or "No summary generated."

    logger.debug(f"SOTA generation complete: {len(text)} characters")
    logger.info(f"SOTA {text}...")
    return {"status": "completed", "text": text}


async def launch_summarizer_server() -> None:
    """Launch the summarizer server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
