"""Automation of dev tasks."""

from typing import Any


def task_format() -> dict[Any, Any]:
    """Format code with ruff."""
    return {
        "actions": ["ruff format ."],
        "verbosity": 2,
    }


def task_check() -> dict[Any, Any]:
    """Lint code with ruff."""
    return {
        "actions": ["ruff check . --fix"],
        "verbosity": 2,
    }


def task_type() -> dict[Any, Any]:
    """Type check with mypy."""
    return {
        "actions": ["mypy ."],
        "verbosity": 2,
    }
