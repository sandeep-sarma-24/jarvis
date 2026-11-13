"""Tool registry: Ollama schemas, argument validation and execution.

Tools are the allowlist: the model (or router) can only name something registered here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("app.tools")

MAX_RESULT_CHARS = 500


class ToolError(Exception):
    """A tool failed. The message is a short sentence that can be spoken as is."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict                      # JSON schema: type=object, properties, required
    run: Callable[..., str]               # sync; returns a speakable sentence, raises ToolError
    info: bool = False                    # result is data for the model (second pass), not speech
    needs_confirmation: bool = False      # v1 registers no such tool; execute() refuses them
    max_chars: int = MAX_RESULT_CHARS     # result cap; screen outlines need more than a sentence


@dataclass(frozen=True)
class ToolResult:
    name: str
    args: dict
    ok: bool
    result: str
    info: bool = False


_TYPES = {"string": str, "integer": int, "boolean": bool}


def _coerce(key: str, spec: dict, value: Any) -> Any:
    kind = spec.get("type", "string")
    if kind == "integer":
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolError(f"{key} must be a whole number.")
    elif kind == "boolean":
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if not isinstance(value, bool):
            raise ToolError(f"{key} must be true or false.")
    elif not isinstance(value, str):
        raise ToolError(f"{key} must be text.")
    if "enum" in spec and value not in spec["enum"]:
        raise ToolError(f"{key} must be one of: {', '.join(map(str, spec['enum']))}.")
    if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
        raise ToolError(f"{key} must be between {spec.get('minimum')} and {spec.get('maximum')}.")
    return value


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}
        self._schemas = [
            {"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}}
            for t in tools
        ]  # built once, stable order: Ollama caches the prompt prefix

    def schemas(self) -> list[dict]:
        return self._schemas

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def validate(self, name: str, args: Any) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"I don't have a tool called {name}.")
        if not isinstance(args, dict):
            raise ToolError("Tool arguments must be an object.")
        props = tool.parameters.get("properties", {})
        if extra := set(args) - set(props):
            raise ToolError(f"Unexpected argument: {sorted(extra)[0]}.")
        if missing := [k for k in tool.parameters.get("required", []) if k not in args]:
            raise ToolError(f"Missing argument: {missing[0]}.")
        return {k: _coerce(k, props[k], v) for k, v in args.items()}

    def execute(self, name: str, args: Any) -> ToolResult:
        """Sync and blocking: call via asyncio.to_thread. Never raises."""
        tool = self._tools.get(name)
        info = bool(tool and tool.info)
        try:
            clean = self.validate(name, args)
            if tool.needs_confirmation:
                raise ToolError("That needs confirmation first.")
            result = tool.run(**clean)[:tool.max_chars]
            ok = True
        except ToolError as exc:
            clean, result, ok = args if isinstance(args, dict) else {}, str(exc), False
        except Exception as exc:  # a tool bug must not break the turn
            logger.exception("tool %s crashed", name)
            clean, result, ok = args if isinstance(args, dict) else {}, f"That failed: {type(exc).__name__}.", False
        logger.info("TOOL %s args=%s ok=%s result=%s", name, clean, ok, result)
        return ToolResult(name, clean, ok, result, info)
