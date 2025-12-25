"""Fixed pipeline orchestrator with LLM validation at each step.

Python manages the pipeline (search → filter → parse → analyze → synthesize).
LLM only validates results and can request redo with new parameters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

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
from sotaforge.utils.constants import MODEL
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

# Prompts moved to sotaforge.utils.prompts

MAX_RETRIES = 3


def _get_last_messages(messages: list[Any], n: int = 5) -> list[Any]:
    """Get last N messages for logging to reduce log verbosity."""
    return messages[-n:] if len(messages) > n else messages


def _normalize_tool_result(result: Any) -> Any:
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


async def _execute_tool_calls(messages: list[Any]) -> list[Any]:
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

        logger.info(f"Found tool calls: {tool_calls}")
        if tool_calls:
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

                logger.debug(
                    "Executing tool: %s with args: %s",
                    tool_name,
                    json.dumps(tool_args),
                )

                # Inject messages context for store_tool_results
                if tool_name == "db_store_tool_results":
                    tool_args["messages"] = messages
                    logger.debug(
                        "Injected messages context into tool args for %s", tool_name
                    )
                    logger.debug("Type of messages: %s", type(tool_args["messages"]))
                    logger.debug(
                        type(tool_args["messages"]),
                    )
                    logger.debug(
                        len(tool_args["messages"]),
                    )
                    logger.debug(type(tool_args["messages"][0]))
                tool = await server.get_tool(tool_name)
                result = await tool.run(tool_args)

                logger.debug(
                    "Tool %s executed with raw result: %s",
                    tool_name,
                    json.dumps(result, default=str),
                )

                # Wrap result with metadata so LLM can see the tool_call_id
                wrapped_result = {
                    "tool_call_id": call_id,
                    "tool_name": tool_name,
                    "result": _normalize_tool_result(result),
                }
                result_content = json.dumps(wrapped_result, default=str)
                logger.debug(
                    "Tool %s executed with normalized result: %s",
                    tool_name,
                    result_content,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_content,
                    }
                )
    return messages


async def validate_step(
    messages: list[Any], prompt: str, openai_tools: list[dict]
) -> tuple[bool, list[Any]]:
    """Ask the LLM to approve or redo the current step.

    Returns a tuple: (approved, messages_with_llm_response, llm_message).
    """
    validation_messages = [
        {"role": "system", "content": VALIDATION_PROMPT},
        {"role": "user", "content": prompt},
    ]
    messages.extend(validation_messages)

    logger.info("Validating step with LLM")
    logger.debug(
        "Messages before validation (last 5): %s",
        json.dumps(_get_last_messages(messages), indent=2, default=str),
    )

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
        logger.debug(f"LLM ({msg.role}) Content: {msg.content}")

    # Execute any tool calls from the validation response
    messages = await _execute_tool_calls(messages)

    # Approve when the assistant explicitly says APPROVE (case-insensitive)
    approved = bool(msg.content and msg.content.strip().upper().startswith("APPROVE"))

    logger.info("Validating step with LLM")
    logger.debug(
        "Messages after validation (last 5): %s",
        json.dumps(_get_last_messages(messages), indent=2, default=str),
    )

    return approved, messages


async def process_message_and_gets_llm_response(
    messages: list[Any],
    openai_tools: list[dict],
) -> list[Any]:
    """Single-pass: execute msg.tool_calls, then fetch next LLM reply."""
    logger.info("Processing message with LLM")
    logger.debug(
        "Messages before processing (last 5): %s",
        json.dumps(_get_last_messages(messages), indent=2, default=str),
    )

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
        logger.debug(f"LLM ({msg.role}) Content: {msg.content}")

    # Execute any tool calls from the LLM response
    messages = await _execute_tool_calls(messages)

    logger.info("Processing message with LLM")
    logger.debug(
        "Messages after processing (last 5): %s",
        json.dumps(_get_last_messages(messages), indent=2, default=str),
    )

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
    ids = db_store.upsert_documents("8_final_sota", [doc])
    return {"collection": "8_final_sota", "stored": True, "ids": ids}


async def run_llm_sota(topic: str) -> int:
    """Run fixed pipeline with LLM validation at each step."""
    logger.info(f"Starting SOTA generation for topic: {topic}")

    messages: list[Any] = [
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

    # Step 1: Search for web and paper sources
    number_of_tries = 1
    approved = False
    messages.append({"role": "user", "content": SEARCH_INSTRUCTION.format(topic=topic)})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    # Ask LLM to persist search results into collections
    messages.append({"role": "user", "content": SAVE_SEARCH_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    while not approved and number_of_tries < MAX_RETRIES:
        logger.debug(
            "messages before search validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )
        approved, messages = await validate_step(
            messages, prompt=VALIDATE_SEARCH, openai_tools=openai_tools
        )
        number_of_tries += 1
        logger.debug(
            "messages after search validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )

    # Step 2: Filter the results
    number_of_tries = 1
    approved = False

    messages.append({"role": "user", "content": FILTER_INSTRUCTION.format(topic=topic)})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    # Ask LLM to persist filtered results into collections
    messages.append({"role": "user", "content": SAVE_FILTER_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    while not approved and number_of_tries < MAX_RETRIES:
        logger.debug(
            "messages before filter validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )
        approved, messages = await validate_step(
            messages, prompt=VALIDATE_FILTER, openai_tools=openai_tools
        )
        number_of_tries += 1
        logger.debug(
            "messages after filter validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )

    # Step 3: Parse selected sources
    number_of_tries = 1
    approved = False
    messages.append({"role": "user", "content": PARSE_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    # Ask LLM to persist parsed documents into collection
    messages.append({"role": "user", "content": SAVE_PARSE_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    while not approved and number_of_tries < MAX_RETRIES:
        logger.debug(
            "messages before parse validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )
        approved, messages = await validate_step(
            messages, prompt=VALIDATE_PARSE, openai_tools=openai_tools
        )
        number_of_tries += 1
        logger.debug(
            "messages after parse validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )

    # Step 4: Analyze parsed content
    number_of_tries = 1
    approved = False
    messages.append({"role": "user", "content": ANALYZE_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    messages.append({"role": "user", "content": SAVE_ANALYZE_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    while not approved and number_of_tries < MAX_RETRIES:
        logger.debug(
            "messages before analyze validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )
        approved, messages = await validate_step(
            messages, prompt=VALIDATE_ANALYZE, openai_tools=openai_tools
        )
        number_of_tries += 1
        logger.debug(
            "messages after analyze validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )

    # Step 5: Synthesize final SOTA report
    number_of_tries = 1
    approved = False
    messages.append({"role": "user", "content": SYNTHESIZE_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    messages.append({"role": "user", "content": SAVE_SYNTHESIZE_INSTRUCTION})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    while not approved and number_of_tries < MAX_RETRIES:
        logger.debug(
            "messages before synthesize validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )
        approved, messages = await validate_step(
            messages, prompt=VALIDATE_SYNTHESIZE, openai_tools=openai_tools
        )
        number_of_tries += 1
        logger.debug(
            "messages after synthesize validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )

    # Step 6: Store final SOTA report
    number_of_tries = 1
    approved = False
    messages.append({"role": "user", "content": STORE_INSTRUCTION.format(topic=topic)})
    messages = await process_message_and_gets_llm_response(
        messages, openai_tools=openai_tools
    )
    while not approved and number_of_tries < MAX_RETRIES:
        logger.debug(
            "messages before store validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
        )
        approved, messages = await validate_step(
            messages, prompt=VALIDATE_STORE, openai_tools=openai_tools
        )
        number_of_tries += 1
        logger.debug(
            "messages after store validate (last 5): %s",
            json.dumps(_get_last_messages(messages), indent=2, default=str),
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
