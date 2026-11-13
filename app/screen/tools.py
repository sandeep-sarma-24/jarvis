"""Model-facing screen tools. Info tool results go back to the model; actions are spoken."""
from __future__ import annotations

from app.screen import core
from app.tools.registry import Tool

_ID = {"type": "string", "description": "element id like e12"}


def _look(app: str | None = None) -> str:
    return core.outline(core.snapshot(app or None))


def _find(query: str) -> str:
    hits = core.find(query)
    return "\n".join(core.line(e, indent=False) for e in hits) or f"No element matches {query}."


def screen_tools() -> list[Tool]:
    return [
        Tool("look_at_screen",
             "Read the UI of a Mac app (default: frontmost) as a list of elements with ids, e.g. buttons, "
             "fields, links. Use before pressing or typing.",
             {"type": "object", "properties": {"app": {"type": "string", "description": "app name, optional"}},
              "required": []}, _look, info=True, max_chars=3000),
        Tool("find_element", "Find elements by name in the last look at the screen.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
             _find, info=True, max_chars=1200),
        Tool("press_element", "Press a button, link or other element by id from the last look.",
             {"type": "object", "properties": {"id": _ID}, "required": ["id"]},
             lambda id: core.act(id, "press")),
        Tool("type_into", "Type text into a text field by id from the last look.",
             {"type": "object", "properties": {"id": _ID, "text": {"type": "string"}},
              "required": ["id", "text"]},
             lambda id, text: core.act(id, "set_value", text)),
    ]
