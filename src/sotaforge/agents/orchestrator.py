"""Fixed pipeline orchestrator with LLM validation at each step.

Python manages the pipeline (search → filter → parse → analyze → synthesize).
LLM only validates results and can request redo with new parameters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from fastmcp import FastMCP
from openai import AsyncOpenAI

from sotaforge.agents import (
    analyzer_server,
    db_agent,
    filter_agent,
    parser_server,
    search_server,
    synthesizer_server,
)
from sotaforge.utils.constants import MODEL, CollectionNames
from sotaforge.utils.dataclasses import Document
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.logger import get_logger
from sotaforge.utils.prompts import (
    ANALYZE_INSTRUCTION,
    FILTER_INSTRUCTION,
    ORCHESTRATOR_SYSTEM_PROMPT,
    PARSE_INSTRUCTION,
    SAVE_ANALYZE_INSTRUCTION,
    SAVE_FILTER_INSTRUCTION,
    SAVE_PARSE_INSTRUCTION,
    SAVE_SEARCH_INSTRUCTION,
    SAVE_SYNTHESIZE_INSTRUCTION,
    SEARCH_INSTRUCTION,
    STORE_INSTRUCTION,
    SYNTHESIZE_INSTRUCTION,
    VALIDATE_ANALYZE,
    VALIDATE_FILTER,
    VALIDATE_PARSE,
    VALIDATE_SEARCH,
    VALIDATE_STORE,
    VALIDATE_SYNTHESIZE,
    VALIDATION_PROMPT,
)
from sotaforge.utils.utils import get_tools_for_openai

logger = get_logger("sotaforge.orchestrator")
llm = AsyncOpenAI()

server = FastMCP("orchestrator")

# Mount in-process FastMCP servers with prefixes
server.mount(search_server.server, prefix="search")
server.mount(filter_agent.server, prefix="filter")
server.mount(parser_server.server, prefix="parser")
server.mount(analyzer_server.server, prefix="analyzer")
server.mount(synthesizer_server.server, prefix="synthesizer")
server.mount(db_agent.server, prefix="db")

db_store = ChromaStore()

# Type alias for message dictionaries
ChatMessage = Dict[str, Any]

MAX_RETRIES = 3


def _get_last_messages(messages: List[ChatMessage], n: int = 5) -> List[ChatMessage]:
    """Get last N messages for logging to reduce log verbosity."""
    return messages[-n:] if len(messages) > n else messages


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


async def validate_step(
    messages: List[ChatMessage], prompt: str, openai_tools: list[dict]
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

    response = await llm.chat.completions.create(
        model=MODEL,  # type: ignore[call-overload]
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

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
    openai_tools: list[dict],
) -> List[ChatMessage]:
    """Single-pass: execute msg.tool_calls, then fetch next LLM reply."""
    logger.info("Processing message with LLM")

    response = await llm.chat.completions.create(
        model=MODEL,  # type: ignore[call-overload]
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    msg_dict = msg.model_dump(exclude_unset=True)
    messages.append(msg_dict)

    if msg.content:
        logger.debug(f"LLM response: {msg.content[:100]}...")

    # Execute any tool calls from the LLM response
    messages = await _execute_tool_calls(messages)

    return messages


@server.tool(
    name="store_final_sota",
    description="Store the final SOTA report text.",
)
async def store_final_sota(sota_text: str, topic: str | None = None) -> Dict[str, Any]:
    """Persist the final SOTA output in Chroma."""
    doc = Document(
        title=topic or "Final SOTA Report",
        text=sota_text,
        source_type="sota_report",
        metadata={"type": "final_sota", "topic": topic}
        if topic
        else {"type": "final_sota"},
    )
    ids = db_store.upsert_documents(CollectionNames.FINAL_SOTA, [doc])
    return {"collection": CollectionNames.FINAL_SOTA, "stored": True, "ids": ids}


async def run_llm_sota(topic: str) -> int:
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

        messages.append({"role": "user", "content": step_instruction})
        await process_message_and_gets_llm_response(messages, openai_tools=openai_tools)

        if save_instruction:  # Only run save if instruction provided
            messages.append({"role": "user", "content": save_instruction})
            await process_message_and_gets_llm_response(
                messages, openai_tools=openai_tools
            )

        while not approved and number_of_tries < MAX_RETRIES:
            logger.debug(
                f"{step_name} validation attempt {number_of_tries}/{MAX_RETRIES}"
            )
            approved, _ = await validate_step(
                messages, prompt=validation_prompt, openai_tools=openai_tools
            )
            number_of_tries += 1

        if not approved:
            logger.warning(
                f"{step_name} step not approved after {MAX_RETRIES} attempts"
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
        PARSE_INSTRUCTION, SAVE_PARSE_INSTRUCTION, VALIDATE_PARSE, "parse"
    )

    # Step 4: Analyze
    await run_pipeline_step(
        ANALYZE_INSTRUCTION, SAVE_ANALYZE_INSTRUCTION, VALIDATE_ANALYZE, "analyze"
    )

    # Step 5: Synthesize
    await run_pipeline_step(
        SYNTHESIZE_INSTRUCTION,
        SAVE_SYNTHESIZE_INSTRUCTION,
        VALIDATE_SYNTHESIZE,
        "synthesize",
    )

    # Step 6: Store final SOTA report
    await run_pipeline_step(
        STORE_INSTRUCTION.format(topic=topic),
        "",  # No save instruction for final step
        VALIDATE_STORE,
        "store",
    )

    logger.info("SOTA generation pipeline completed successfully")
    return 0


async def _run_server() -> int:
    """Run the orchestrator MCP server exposing mounted tools."""
    await server.run_stdio_async()
    return 0


def main() -> int:
    """Entry point for the orchestrator server."""
    return asyncio.run(_run_server())


if __name__ == "__main__":
    raise SystemExit(main())
