"""Centralized LLM client configuration for SOTAforge.

This module provides a single AsyncOpenAI instance that is shared across
all modules in the application. This approach offers several benefits:

1. Resource Efficiency: Reuses HTTP connection pools
2. Configuration Management: Single source of truth for API settings
3. Testing: Easy to mock in one place
4. Monitoring: Single point for usage tracking

Usage:
    # For direct OpenAI API calls
    from sotaforge.utils.llm import get_llm
    llm = get_llm()
    response = await llm.chat.completions.create(...)

    # For PydanticAI agents
    from sotaforge.utils.llm import get_pydantic_model
    agent = Agent(get_pydantic_model(), output_type=MyType, ...)
"""

import os
from functools import lru_cache

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from sotaforge.utils.constants import MODEL
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> AsyncOpenAI:
    """Get the singleton AsyncOpenAI client instance.

    Uses @lru_cache to ensure only one instance is created and reused.
    This is thread-safe and cleaner than using global variables.

    Returns:
        Configured AsyncOpenAI client

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set

    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set. "
            "Create a .env.secrets file with your API key."
        )

    logger.info("Initialized AsyncOpenAI client")
    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def get_pydantic_model(model_name: str = MODEL) -> OpenAIChatModel:
    """Get a PydanticAI OpenAIChatModel using the shared client.

    This combines the shared AsyncOpenAI client with a specific model name
    for use with PydanticAI agents.

    Args:
        model_name: The OpenAI model to use (default: MODEL from constants)

    Returns:
        Configured OpenAIChatModel for PydanticAI

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set

    """
    client = get_llm()
    # Use the official OpenAIProvider to wrap the shared client
    provider = OpenAIProvider(openai_client=client)
    logger.info(f"Initialized PydanticAI model: {model_name}")
    return OpenAIChatModel(model_name, provider=provider)


def reset_llm() -> None:
    """Clear the cached LLM instances (mainly for testing purposes)."""
    get_llm.cache_clear()
    get_pydantic_model.cache_clear()
