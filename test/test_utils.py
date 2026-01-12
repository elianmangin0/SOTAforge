"""Unit tests for utils.py functions."""

from typing import Any

import pytest

from sotaforge.utils.utils import get_tools_for_openai


class TestGetToolsForOpenAI:
    """Tests for get_tools_for_openai function."""

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_all_tools(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test getting all tools without filtering."""
        tools = await get_tools_for_openai(mock_fastmcp_server)

        assert len(tools) == 3
        assert all(tool["type"] == "function" for tool in tools)

        # Check tool names are converted (dots to underscores)
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "search_query" in tool_names
        assert "parse_document" in tool_names
        assert "internal_debug" in tool_names

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_with_prefix_filter(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test filtering tools by allowed prefixes."""
        tools = await get_tools_for_openai(
            mock_fastmcp_server, allowed_prefixes=["search"]
        )

        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search_query"

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_multiple_prefixes(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test filtering with multiple allowed prefixes."""
        tools = await get_tools_for_openai(
            mock_fastmcp_server, allowed_prefixes=["search", "parse"]
        )

        assert len(tools) == 2
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "search_query" in tool_names
        assert "parse_document" in tool_names
        assert "internal_debug" not in tool_names

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_structure(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test the structure of returned OpenAI tools."""
        tools = await get_tools_for_openai(
            mock_fastmcp_server, allowed_prefixes=["search"]
        )

        tool = tools[0]
        assert tool["type"] == "function"
        assert "function" in tool

        function = tool["function"]
        assert "name" in function
        assert "description" in function
        assert "parameters" in function

        # Check parameters structure
        params = function["parameters"]
        assert params["type"] == "object"
        assert "properties" in params

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_empty_prefix_filter(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test with empty allowed_prefixes list returns all tools."""
        tools = await get_tools_for_openai(mock_fastmcp_server, allowed_prefixes=[])

        # Empty prefix list means no filtering - returns all tools
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_nonexistent_prefix(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test filtering with prefix that doesn't match any tools."""
        tools = await get_tools_for_openai(
            mock_fastmcp_server, allowed_prefixes=["nonexistent"]
        )

        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_get_tools_for_openai_default_parameters(
        self, mock_fastmcp_server: Any
    ) -> None:
        """Test that tools without parameters get default type object."""
        tools = await get_tools_for_openai(
            mock_fastmcp_server, allowed_prefixes=["parse"]
        )

        assert len(tools) == 1
        params = tools[0]["function"]["parameters"]
        assert params["type"] == "object"
        # The actual implementation may not include empty properties/required
