"""Tool registry for dynamic tool management."""

import asyncio
import logging
from typing import Any

from miniclaw.agent.tools.base import Tool
from miniclaw.config.schema import ToolRetryConfig

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(self, retry_config: ToolRetryConfig | None=None):
        self._tools: dict[str, Tool] = {}
        self._retry_config = retry_config

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def get_definitions(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def _should_retry(self, name: str) -> bool:
        cfg = self._retry_config
        if cfg is None or not cfg.enabled or cfg.retry_attempts < 1:
            return False
        if cfg.retry_tools:
            return name in cfg.retry_tools
        return True

    async def _execute_with_retry(self, tool: Tool, params: dict[str, Any]) -> str:
        cfg = self._retry_config
        max_attempts = max(1, cfg.retry_attempts)
        backoff = max(0.0, cfg.retry_backoff_seconds)
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return await tool.execute(**params)
            except asyncio.TimeoutError as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                logger.warning(
                    "Tool '%s' timed out (attempt %d/%d), retrying in %.1fs",
                    tool.name, attempt, max_attempts, backoff,
                )
                if backoff > 0:
                    await asyncio.sleep(backoff)
                backoff = min(
                    max(0.0, cfg.retry_max_backoff_seconds),
                    backoff * max(1.0, cfg.retry_backoff_multiplier),
                )

        raise last_exc

    async def execute(self, name: str, params: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        if not isinstance(params, dict):
            return (
                f"Error: Invalid parameters for tool '{name}': arguments must be an object. "
                "This may happen when tool-call arguments are truncated. "
                "Please retry with shorter content or split the task into smaller tool calls."
            )

        parse_error = params.get("__miniclaw_tool_args_error__")
        if parse_error:
            detail = params.get("__miniclaw_tool_args_error_msg__", "unknown parse error")
            return (
                f"Error: Tool-call arguments for '{name}' could not be parsed ({detail}). "
                "This usually means the arguments were truncated. "
                "Please reduce argument size and retry. For long file output, use write_file for the first chunk and append_file for subsequent chunks."
            )

        try:
            errors = tool.validate_params(params)
            if errors:
                if name in {"write_file", "append_file"} and "missing required content" in "; ".join(errors):
                    return (
                        f"Error: Invalid parameters for tool '{name}': missing required content. "
                        "This often indicates argument truncation. Please shorten each content chunk and retry; "
                        "for large files, write in multiple append_file chunks."
                    )
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)

            if self._should_retry(name):
                return await self._execute_with_retry(tool, params)
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
