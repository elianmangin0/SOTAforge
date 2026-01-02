"""Fixed pipeline orchestrator with LLM validation at each step.

Python manages the pipeline (search → filter → parse → analyze → synthesize).
LLM only validates results and can request redo with new parameters.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from openai import AsyncOpenAI, RateLimitError

from sotaforge.agents import (
    analyzer_server,
    db_agent,
    filter_agent,
    parser_server,
    search_server,
    synthesizer_server,
)
from sotaforge.utils.constants import (
    MAX_MESSAGE_HISTORY,
    MAX_RETRIES,
    MODEL,
)
from sotaforge.utils.logger import get_logger
from sotaforge.utils.prompts import (
    ANALYZE_INSTRUCTION,
    FILTER_INSTRUCTION,
    ORCHESTRATOR_SYSTEM_PROMPT,
    PARSE_INSTRUCTION,
    SAVE_FILTER_INSTRUCTION,
    SAVE_SEARCH_INSTRUCTION,
    SAVE_SYNTHESIZE_INSTRUCTION,
    SEARCH_INSTRUCTION,
    SYNTHESIZE_INSTRUCTION,
    VALIDATE_ANALYZE,
    VALIDATE_FILTER,
    VALIDATE_PARSE,
    VALIDATE_SEARCH,
    VALIDATE_SYNTHESIZE,
    VALIDATION_PROMPT,
)
from sotaforge.utils.utils import get_tools_for_openai

logger = get_logger("sotaforge.orchestrator")
llm = AsyncOpenAI()

server = FastMCP("orchestrator")

# Global progress queue for streaming updates (injected by API)
progress_queue: Optional[asyncio.Queue[Dict[str, Any]]] = None


async def emit_progress(status: str, message: str, step: str = "") -> None:
    """Emit progress update to the queue if available.

    Args:
        status: Current status (searching, filtering, parsing,
            analyzing, synthesizing, etc.)
        message: Human-readable message
        step: Current pipeline step

    """
    if progress_queue:
        await progress_queue.put(
            {
                "status": status,
                "message": message,
                "step": step,
                "timestamp": datetime.now().isoformat(),
            }
        )


async def _emit_tool_progress(tool_name: str, tool_args: dict[str, Any]) -> None:
    """Emit detailed progress for specific tool executions.

    Args:
        tool_name: Name of the tool being executed
        tool_args: Arguments passed to the tool

    """
    if not progress_queue:
        return

    # Extract step from tool name prefix
    step = tool_name.split("_")[0] if "_" in tool_name else "processing"
    logger.debug(f"Emitting progress for tool: {tool_name} in step: {step}")

    # Search tools
    if "search_web" in tool_name:
        query = tool_args.get("query", "")
        await emit_progress(step, f"Searching web for: {query}", step)
    elif "search_papers" in tool_name:
        query = tool_args.get("query", "")
        await emit_progress(step, f"Searching papers for: {query}", step)

    # Filter tools
    elif "filter_results" in tool_name:
        collection = tool_args.get("collection", "")
        if collection:
            try:
                from sotaforge.utils.db import ChromaStore

                db = ChromaStore()
                docs = db.fetch_documents(collection)
                await emit_progress(
                    step,
                    f"Retrieved {len(docs)} documents from collection: {collection}",
                    step,
                )
                for doc in docs:
                    doc_dict = (
                        doc
                        if isinstance(doc, dict)
                        else (doc.to_dict() if hasattr(doc, "to_dict") else {})
                    )
                    title = doc_dict.get("title", "Unknown")
                    await emit_progress(step, f"Scoring: {title[:60]}...", step)
            except Exception as e:
                logger.warning(f"Failed to retrieve documents for progress: {e}")

    # Parser tools
    elif "parse_documents" in tool_name:
        collection = tool_args.get("collection", "")
        if collection:
            try:
                from sotaforge.utils.db import ChromaStore

                db = ChromaStore()
                docs = db.fetch_documents(collection)
                await emit_progress(
                    step, f"Retrieved {len(docs)} documents from: {collection}", step
                )
                for doc in docs:
                    doc_dict = (
                        doc
                        if isinstance(doc, dict)
                        else (doc.to_dict() if hasattr(doc, "to_dict") else {})
                    )
                    title = doc_dict.get("title", "Unknown")
                    url = doc_dict.get("url", "")
                    await emit_progress(
                        step, f"Parsing: {title[:50]}... ({url[:40]})", step
                    )
            except Exception as e:
                logger.warning(f"Failed to retrieve documents for progress: {e}")

    # Analyzer tools
    elif "analyze_documents" in tool_name:
        collection = tool_args.get("collection", "")
        if collection:
            try:
                from sotaforge.utils.db import ChromaStore

                db = ChromaStore()
                docs = db.fetch_documents(collection)
                await emit_progress(
                    step, f"Retrieved {len(docs)} documents from: {collection}", step
                )
                for doc in docs:
                    doc_dict = (
                        doc
                        if isinstance(doc, dict)
                        else (doc.to_dict() if hasattr(doc, "to_dict") else {})
                    )
                    title = doc_dict.get("title", "Unknown")
                    await emit_progress(step, f"Analyzing: {title[:60]}...", step)
            except Exception as e:
                logger.warning(f"Failed to retrieve documents for progress: {e}")

    # Synthesizer tools
    elif "write_sota" in tool_name:
        await emit_progress(step, "Writing SOTA summary...", step)

    # Database operations
    elif "db_store" in tool_name or "db_save" in tool_name:
        await emit_progress(step, "Saving results to database...", step)
    elif "db_retrieve" in tool_name or "db_get" in tool_name:
        await emit_progress(step, "Retrieving data from database...", step)


# Mount in-process FastMCP servers with prefixes
server.mount(search_server.server, prefix="search")
server.mount(filter_agent.server, prefix="filter")
server.mount(parser_server.server, prefix="parser")
server.mount(analyzer_server.server, prefix="analyzer")
server.mount(synthesizer_server.server, prefix="synthesizer")
server.mount(db_agent.server, prefix="db")

# Type alias for message dictionaries
ChatMessage = Dict[str, Any]


async def _chat_with_rate_limit_retry(
    messages: List[ChatMessage], openai_tools: list[dict[str, Any]]
) -> Any:
    """Call OpenAI chat; on retryable rate-limit, wait 60s then retry once.

    Non-retryable errors (request too large) are raised immediately.
    Trims message history before sending to prevent token limit issues.
    """
    # Trim message history to prevent token limit errors

    try:
        return await llm.chat.completions.create(
            model=MODEL,  # type: ignore[call-overload]
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
    except RateLimitError as e:
        error_message = str(e)

        # Check if this is a "request too large" error (non-retryable)
        if "Request too large" in error_message or "must be reduced" in error_message:
            logger.warning(
                "OpenAI rate limit hit (TPM/RPM). Cooling down 60s before retry: %s",
                e,
            )
            await asyncio.sleep(60)
            try:
                return await llm.chat.completions.create(
                    model=MODEL,  # type: ignore[call-overload]
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
            except RateLimitError as e2:
                logger.error("Retry after cooldown also hit rate limit: %s", e2)
                logger.warning(
                    "Reducing message history and retrying once more after cooldown"
                )
                await asyncio.sleep(60)
                # Trim message history further and retry once more
                trimmed_messages = _trim_message_history(messages, max_messages=5)
                return await llm.chat.completions.create(
                    model=MODEL,  # type: ignore[call-overload]
                    messages=trimmed_messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
        else:
            logger.error("Non-retryable OpenAI rate limit error: %s", e)
            raise


def _get_last_messages(messages: List[ChatMessage], n: int = 5) -> List[ChatMessage]:
    """Get last N messages for logging to reduce log verbosity."""
    return messages[-n:] if len(messages) > n else messages


def _trim_message_history(
    messages: List[ChatMessage], max_messages: int = MAX_MESSAGE_HISTORY
) -> List[ChatMessage]:
    """Trim message history to prevent token limit errors.

    Keeps the most recent messages and ensures we don't cut between
    tool_calls and their responses.

    Args:
        messages: Full message history
        max_messages: Maximum number of messages to keep

    Returns:
        Trimmed message list

    """
    if len(messages) <= max_messages:
        return messages

    # Take the last max_messages
    start_idx = len(messages) - max_messages
    recent_messages = messages[start_idx:]

    # If first message is a tool response, back up to include its tool_call
    while (
        recent_messages and recent_messages[0].get("role") == "tool" and start_idx > 0
    ):
        start_idx -= 1
        recent_messages = messages[start_idx:]

    logger.info(
        "Trimmed message history from %s to %s messages",
        len(messages),
        len(recent_messages),
    )

    return recent_messages


def _normalize_tool_result(result: Any) -> Any:
    """Normalize tool result from various formats to a consistent structure."""
    # 1. Unwrap SDK result
    content = result.content if hasattr(result, "content") else result

    # 2. If it's a list with a text object, extract text
    if isinstance(content, list) and len(content) == 1:
        item = content[0]

        # SDK-style text object
        if hasattr(item, "text"):
            content = item.text

    # 3. If it's a JSON string, parse it
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass  # leave as-is if not JSON

    return content


async def _execute_tool_calls(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Execute any pending tool calls on the last message and append tool responses.

    Args:
        messages: The message history

    Returns:
        Updated messages with tool responses appended

    """
    if messages:
        last = messages[-1]
        # Handle both dict and object types
        if isinstance(last, dict):
            tool_calls = last.get("tool_calls")
        else:
            tool_calls = getattr(last, "tool_calls", None)

        if tool_calls:
            logger.info(f"Executing {len(tool_calls)} tool call(s)")
            for call in tool_calls:
                # Handle both dict and object types
                if isinstance(call, dict):
                    tool_name = call["function"]["name"]
                    tool_args = json.loads(call["function"]["arguments"])
                    call_id = call["id"]
                else:
                    tool_name = call.function.name
                    tool_args = json.loads(call.function.arguments)
                    call_id = call.id

                logger.debug(f"Executing tool: {tool_name}")

                # Emit detailed progress for specific tools
                await _emit_tool_progress(tool_name, tool_args)

                # Inject messages context for store_tool_results
                if tool_name == "db_store_tool_results":
                    tool_args["messages"] = messages
                    logger.debug("Injected messages context for store_tool_results")

                tool = await server.get_tool(tool_name)
                result = await tool.run(tool_args)

                # Wrap result with metadata so LLM can see the tool_call_id
                wrapped_result = {
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "result": _normalize_tool_result(result),
                }
                result_content = json.dumps(wrapped_result, default=str)
                logger.debug(f"Tool {tool_name} completed")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_content,
                    }
                )
    return messages


def _extract_synthesized_sota_text(messages: List[ChatMessage]) -> str:
    """Extract the synthesized SOTA text from tool results.

    Looks for the latest tool message produced by the synthesizer's `write_sota`
    tool and returns its `text` field. Falls back to empty string if not found.
    """
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool":
            payload = msg.get("content")
            try:
                data = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:
                continue

            tool_name = (data or {}).get("tool_name")
            if not tool_name:
                continue

            # Accept both prefixed and unprefixed names
            if tool_name == "synthesizer_write_sota" or tool_name.endswith(
                "write_sota"
            ):
                result = (data or {}).get("result")
                if isinstance(result, dict):
                    text = result.get("text") or result.get("sota")
                    if isinstance(text, str):
                        return text

    return ""


async def validate_step(
    messages: List[ChatMessage], prompt: str, openai_tools: list[dict[str, Any]]
) -> tuple[bool, List[ChatMessage]]:
    """Ask the LLM to approve or redo the current step.

    Returns a tuple: (approved, messages_with_llm_response).
    """
    validation_messages = [
        {"role": "system", "content": VALIDATION_PROMPT},
        {"role": "user", "content": prompt},
    ]
    messages.extend(validation_messages)

    logger.info("Validating step with LLM")

    response = await _chat_with_rate_limit_retry(messages, openai_tools)

    msg = response.choices[0].message
    msg_dict = msg.model_dump(exclude_unset=True)
    messages.append(msg_dict)

    if msg.content:
        logger.debug(f"LLM validation response: {msg.content[:100]}...")

    # Execute any tool calls from the validation response
    messages = await _execute_tool_calls(messages)

    # Approve when the assistant explicitly says APPROVE (case-insensitive)
    approved = bool(msg.content and msg.content.strip().upper().startswith("APPROVE"))

    logger.info(f"Step validation: {'APPROVED' if approved else 'NEEDS RETRY'}")

    return approved, messages


async def process_message_and_gets_llm_response(
    messages: List[ChatMessage],
    openai_tools: list[dict[str, Any]],
) -> List[ChatMessage]:
    """Single-pass: execute msg.tool_calls, then fetch next LLM reply."""
    logger.info("Processing message with LLM")

    response = await _chat_with_rate_limit_retry(messages, openai_tools)

    msg = response.choices[0].message
    msg_dict = msg.model_dump(exclude_unset=True)
    messages.append(msg_dict)

    if msg.content:
        logger.debug(f"LLM response: {msg.content[:100]}...")

    # Execute any tool calls from the LLM response
    messages = await _execute_tool_calls(messages)

    return messages


async def run_llm_sota(topic: str) -> dict[str, Any]:
    """Run fixed pipeline with LLM validation at each step."""
    logger.info(f"Starting SOTA generation for topic: {topic}")

    messages: List[ChatMessage] = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
    ]

    openai_tools = await get_tools_for_openai(
        server,
        allowed_prefixes=(
            "search_",
            "filter_",
            "parser_",
            "analyzer_",
            "synthesizer_",
            "db_",
            "store_",
        ),
    )

    async def run_pipeline_step(
        step_instruction: str,
        save_instruction: str,
        validation_prompt: str,
        step_name: str,
    ) -> None:
        """Execute a pipeline step with validation retry logic.

        Args:
            step_instruction: Instruction for the step
            save_instruction: Instruction to save results
            validation_prompt: Validation prompt
            step_name: Name for logging

        """
        approved = False
        number_of_tries = 1

        # Emit progress at start of step
        await emit_progress(step_name, f"Starting {step_name} step...", step_name)

        messages.append({"role": "user", "content": step_instruction})
        await process_message_and_gets_llm_response(messages, openai_tools=openai_tools)

        if save_instruction:  # Only run save if instruction provided
            await emit_progress(step_name, f"Saving {step_name} results...", step_name)
            messages.append({"role": "user", "content": save_instruction})
            await process_message_and_gets_llm_response(
                messages, openai_tools=openai_tools
            )

        while not approved and number_of_tries < MAX_RETRIES:
            logger.debug(
                f"{step_name} validation attempt {number_of_tries}/{MAX_RETRIES}"
            )
            await emit_progress(
                step_name,
                (
                    f"Validating {step_name} results "
                    f"(attempt {number_of_tries}/{MAX_RETRIES})..."
                ),
                step_name,
            )
            approved, _ = await validate_step(
                messages, prompt=validation_prompt, openai_tools=openai_tools
            )
            number_of_tries += 1

        if approved:
            await emit_progress(
                step_name,
                f"✓ {step_name.capitalize()} step completed successfully",
                step_name,
            )
        else:
            logger.warning(
                f"{step_name} step not approved after {MAX_RETRIES} attempts"
            )
            await emit_progress(
                step_name,
                f"⚠ {step_name.capitalize()} step completed with warnings",
                step_name,
            )

    # Step 1: Search
    await run_pipeline_step(
        SEARCH_INSTRUCTION.format(topic=topic),
        SAVE_SEARCH_INSTRUCTION,
        VALIDATE_SEARCH,
        "search",
    )

    # Step 2: Filter
    await run_pipeline_step(
        FILTER_INSTRUCTION.format(topic=topic),
        SAVE_FILTER_INSTRUCTION,
        VALIDATE_FILTER,
        "filter",
    )

    # Step 3: Parse
    await run_pipeline_step(
        PARSE_INSTRUCTION,
        "",  # storage handled automatically by parser tool
        VALIDATE_PARSE,
        "parse",
    )

    # Step 4: Analyze
    await run_pipeline_step(
        ANALYZE_INSTRUCTION,
        "",  # storage handled automatically by analyzer tool
        VALIDATE_ANALYZE,
        "analyze",
    )

    # Step 5: Synthesize
    await run_pipeline_step(
        SYNTHESIZE_INSTRUCTION,
        SAVE_SYNTHESIZE_INSTRUCTION,
        VALIDATE_SYNTHESIZE,
        "synthesize",
    )

    # Return the synthesized SOTA text from the tool result without storing
    logger.info("Extracting synthesized SOTA text from tool results")
    sota_text = _extract_synthesized_sota_text(messages)
    if not sota_text:
        logger.warning(
            "No synthesized SOTA text found in tool results; returning empty text"
        )

    logger.info("SOTA generation pipeline completed successfully (no store)")
    return {"topic": topic, "status": "completed", "text": sota_text}


async def _run_server() -> int:
    """Run the orchestrator MCP server exposing mounted tools."""
    await server.run_stdio_async()
    return 0


def main() -> int:
    """Entry point for the orchestrator server."""
    return asyncio.run(_run_server())


if __name__ == "__main__":
    raise SystemExit(main())
