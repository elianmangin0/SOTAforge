"""Utility functions for SOTAforge."""

from typing import Iterable, List, Optional

from fastmcp import FastMCP


async def get_tools_for_openai(
    server: FastMCP, allowed_prefixes: Optional[Iterable[str]] = None
) -> list:
    """Convert MCP server tools to OpenAI tool format.

    Args:
        server: FastMCP instance exposing tools.
        allowed_prefixes: Optional iterable of prefixes to include. When provided,
            only tools whose names start with one of these prefixes are exposed to
            the LLM. This keeps heavy or low-level tools out of the model context.

    """
    tools = await server.get_tools()
    allowed_list: List[str] = list(allowed_prefixes) if allowed_prefixes else []

    openai_tools = []
    for tool_name, tool in tools.items():
        if allowed_list and not any(tool_name.startswith(p) for p in allowed_list):
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name.replace(".", "_"),
                    "description": tool.description or "No description provided",
                    "parameters": tool.parameters
                    if tool.parameters
                    else {"type": "object", "properties": {}, "required": []},
                },
            }
        )
    return openai_tools
