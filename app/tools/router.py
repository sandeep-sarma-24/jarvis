"""Deterministic phrase router: common commands skip the LLM entirely.

route(text) -> (tool, args) or None. Anchored and strict; anything with a conjunction or a
question falls through to the model.
"""
from __future__ import annotations

import re

_PANES = {
    "wifi": "wifi", "wi fi": "wifi", "wi-fi": "wifi", "bluetooth": "bluetooth", "network": "network",
    "notifications": "notifications", "sound": "sound", "sounds": "sound", "audio": "sound",
    "focus": "focus", "display": "displays", "displays": "displays", "screen": "displays",
    "wallpaper": "wallpaper", "battery": "battery", "lock screen": "lock_screen",
    "control center": "control_center", "keyboard": "keyboard", "trackpad": "trackpad",
    "general": "general", "software update": "software_update", "updates": "software_update",
    "storage": "storage", "accessibility": "accessibility", "privacy": "privacy",
    "security": "privacy", "privacy and security": "privacy",
}
_DOMAIN = r"(?:https?://\S+|[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|app|ai|co|edu|gov)(?:/\S*)?)"

_RULES: list[tuple[re.Pattern, str]] = [(re.compile(p), k) for p, k in [
    (r"(?:search|google)(?: google| the web| the internet)?(?: for)? (?P<q>.+)", "search"),
    (r"(?:open|show|go to) (?:the |my )?(?P<p>.+?) (?:settings|preferences)", "settings"),
    (r"(?:open|show|go to) (?:the |my )?(?P<f>downloads|documents|desktop|home)(?: folder)?", "folder"),
    (r"(?:open|go to) (?P<u>" + _DOMAIN + ")", "url"),
    (r"(?:set |turn )?(?:the )?volume (?:to |at )?(?P<n>\d{1,3})(?: percent| %)?", "volume"),
    (r"(?:turn |set )?(?:the )?volume (?P<d>up|down)", "volume_delta"),
    (r"(?P<m>mute|unmute)(?: the)?(?: volume| sound| audio| mac)?", "mute"),
    (r"(?P<a>pause|play|resume)(?: the)? (?:music|song|spotify|playback)|(?P<a2>pause)", "media"),
    (r"(?P<a3>next|skip|previous|last)(?: the| a)? (?:song|track)", "media"),
    # pronouns ("look at this/me") are camera vision, not the screen
    (r"look at (?:the |my )?(?:screen|window|(?!(?:this|that|it|me|him|her|them|these|those|here)\b)(?P<app>.+?)(?: app| window| screen)?)", "screen"),
    (r"(?:please )?(?:open|launch|start|switch to) (?:the )?(?P<app>.+?)(?: app)?", "app"),
]]


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"^(?:hey |ok |okay )?jarvis\b[,.!:]?\s*", "", t)
    t = re.sub(r"^(?:please|can you|could you|would you)\s+", "", t)
    t = re.sub(r"[\s.!?,]+$", "", t)
    return t


def route(text: str) -> tuple[str, dict] | None:
    t = normalize(text)
    if not t or re.search(r"\b(?:and|then|why|what|how|who)\b", t) and not t.startswith(("search", "google")):
        return None
    for rx, kind in _RULES:
        m = rx.fullmatch(t)
        if not m:
            continue
        g = m.groupdict()
        if kind == "search":
            return "open_url", {"query": g["q"]}
        if kind == "settings":
            pane = _PANES.get(g["p"])
            if pane is None:
                continue  # "open system settings" is an app, not a pane
            return "open_settings", {"pane": pane}
        if kind == "folder":
            return "open_path", {"path": g["f"]}
        if kind == "url":
            u = g["u"]
            return "open_url", {"url": u if u.startswith("http") else "https://" + u}
        if kind == "volume":
            return "set_volume", {"level": int(g["n"])}
        if kind == "volume_delta":
            return "set_volume", {"delta": 10 if g["d"] == "up" else -10}
        if kind == "mute":
            return "set_volume", {"mute": g["m"] == "mute"}
        if kind == "media":
            a = g.get("a") or g.get("a2") or g["a3"]
            return "media_control", {"action": {"resume": "play", "last": "previous", "skip": "next"}.get(a, a)}
        if kind == "screen":
            return "look_at_screen", {"app": g["app"]} if g["app"] else {}
        return "open_app", {"name": g["app"]}
    return None
