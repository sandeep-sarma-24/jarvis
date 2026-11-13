"""Platform-independent screen core: bounded accessibility-tree walk, pruning, id cache, actions.

The backend (app.screen.ax.AXBackend, or a fake in tests) owns every OS call; handles are opaque.
"""
from __future__ import annotations

import difflib
import logging
import threading
import time
from typing import Callable

from app.tools.registry import ToolError

logger = logging.getLogger("app.tools.screen")

NOT_TRUSTED = "I need Accessibility permission. Enable it in System Settings, Privacy and Security, Accessibility."

MAX_DEPTH, MAX_VISITED, TIME_BUDGET, MAX_ELEMENTS = 30, 2000, 3.0, 400
MAX_TYPED = 2000

INTERACTIVE = {
    "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXComboBox",
    "AXTextField", "AXSecureTextField", "AXTextArea", "AXSearchField", "AXLink", "AXMenuItem",
    "AXMenuBarItem", "AXSlider", "AXTab", "AXDisclosureTriangle", "AXIncrementor",
}
CONTEXT = {"AXWindow", "AXSheet", "AXDialog", "AXWebArea", "AXToolbar", "AXTabGroup", "AXHeading"}
NEEDS_LABEL = {"AXGroup", "AXList", "AXTable", "AXOutline"}
TEXT_LIKE = {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"}
ACTION_NAMES = {"AXPress": "press", "AXShowMenu": "show_menu", "AXConfirm": "confirm",
                "AXIncrement": "increment", "AXDecrement": "decrement", "AXRaise": "raise"}
SECURE = "AXSecureTextField"


class ScreenError(ToolError):
    """Screen read/act failed. Message is a short speakable sentence."""


_backend = None
_lock = threading.Lock()
_cache: dict[str, tuple[object, str, str]] = {}   # id -> (handle, raw role, label)
_last: dict | None = None
_subs: list[Callable[[dict], None]] = []


def set_backend(backend) -> None:
    global _backend
    _backend = backend


def _be():
    global _backend
    if _backend is None:
        from app.screen.ax import AXBackend  # lazy: PyObjC only exists on macOS
        _backend = AXBackend()
    return _backend


def _trusted_backend():
    b = _be()
    if not b.trusted():
        raise ScreenError(NOT_TRUSTED)
    return b


def _text(v, n: int) -> str:
    return " ".join(str(v).split())[:n] if v is not None else ""


def snapshot(app: str | None = None) -> dict:
    global _last
    b = _trusted_backend()
    t0 = time.monotonic()
    name, pid, root = b.app_ref(app)
    root_el = {"id": "e0", "parent": None, "depth": 0, "role": "application", "label": _text(name, 80),
               "value": None, "frame": None, "enabled": True, "focused": False, "actions": [],
               "interactive": False, "secure": False}
    els, handles = [root_el], {"e0": (root, "AXApplication", root_el["label"])}
    st = {"visited": 0, "truncated": False}
    deadline = t0 + TIME_BUDGET

    def visit(el, depth: int, par: dict, kdepth: int) -> None:
        if st["truncated"]:
            return
        if depth > MAX_DEPTH or st["visited"] >= MAX_VISITED or time.monotonic() > deadline:
            st["truncated"] = True
            return
        st["visited"] += 1
        try:
            a = b.attrs(el)
        except ScreenError:
            return  # stale or unreadable: skip the subtree, keep the snapshot
        role = a.get("role") or ""
        if role == "AXMenu":
            return  # closed menus are zero-size noise; menu bar items stay pressable
        e = _element(a, role, par)
        if e is not None:
            if len(els) >= MAX_ELEMENTS:
                st["truncated"] = True
                return
            e["id"], e["parent"], e["depth"] = f"e{len(els)}", par["id"], kdepth + 1
            els.append(e)
            handles[e["id"]] = (el, role, e["label"])
            par, kdepth = e, kdepth + 1
        try:
            kids = b.children(el)
        except ScreenError:
            return
        for k in kids:
            visit(k, depth + 1, par, kdepth)

    for k in b.children(root):  # root failure propagates
        visit(k, 1, root_el, 0)

    snap = {"app": root_el["label"], "pid": pid, "ts": time.time(), "truncated": st["truncated"],
            "elements": els}
    with _lock:
        _cache.clear()
        _cache.update(handles)
        _last = snap
        subs = list(_subs)
    logger.info("SNAPSHOT app=%s elements=%d truncated=%s ms=%d", snap["app"], len(els),
                snap["truncated"], (time.monotonic() - t0) * 1000)
    for fn in subs:
        try:
            fn(snap)
        except Exception:
            logger.exception("screen subscriber failed")
    return snap


def _element(a: dict, role: str, par: dict) -> dict | None:
    """Return the element dict for a node worth keeping, else None."""
    secure = role == SECURE or a.get("subrole") == SECURE
    label = next((_text(a.get(k), 80) for k in ("title", "description", "placeholder", "help")
                  if _text(a.get(k), 80)), "")
    value = "•••" if secure else (_text(a.get("value"), 120) or None)
    actions = a.get("actions") or []
    interactive = role in INTERACTIVE or "AXPress" in actions
    if role == "AXStaticText":
        text = label or value or ""
        if not text or text == par["label"]:
            return None
        label, value = text[:80], None
    elif not (interactive or role in CONTEXT or (role == "AXImage" and label)
              or (role in NEEDS_LABEL and label)):
        return None
    frame = a.get("frame")
    if frame and (frame[2] <= 0 or frame[3] <= 0) and role != "AXWindow":
        return None
    return {"role": ("secure" if secure else role[2:].lower() if role.startswith("AX") else role.lower()),
            "label": label, "value": value, "frame": [int(v) for v in frame] if frame else None,
            "enabled": bool(a.get("enabled", True)), "focused": bool(a.get("focused", False)),
            "actions": [ACTION_NAMES[x] for x in actions if x in ACTION_NAMES],
            "interactive": interactive, "secure": secure}


def last() -> dict | None:
    return _last


def subscribe(fn: Callable[[dict], None]) -> Callable[[], None]:
    with _lock:
        _subs.append(fn)

    def unsubscribe() -> None:
        with _lock:
            if fn in _subs:
                _subs.remove(fn)
    return unsubscribe


def act(id: str, action: str, value: str | None = None) -> str:
    b = _trusted_backend()
    with _lock:
        entry = _cache.get(id)
    if entry is None:
        raise ScreenError(f"I don't know element {id}. Look at the screen again.")
    handle, role, label = entry
    gone = ScreenError(f"Element {id} is gone. Look at the screen again.")
    try:
        a = b.attrs(handle)
    except ScreenError:
        raise gone from None
    if a.get("role") != role:
        raise gone
    if role == SECURE or a.get("subrole") == SECURE:
        raise ScreenError("I won't touch password fields.")
    name = label or id
    if not a.get("enabled", True):
        raise ScreenError(f"{name} is disabled.")
    actions = a.get("actions") or []
    if action == "press":
        use = "AXPress" if "AXPress" in actions else next(
            (x for x in ("AXConfirm", "AXPick", "AXOpen") if x in actions), None)
        if use is None:
            raise ScreenError("That can't be pressed.")
        b.perform(handle, use)
        done = f"Pressed {name}."
    elif action == "focus":
        b.set_attr(handle, "AXFocused", True)
        done = f"Focused {name}."
    elif action == "set_value":
        if role not in TEXT_LIKE:
            raise ScreenError("That isn't a text field.")
        if not isinstance(value, str):
            raise ScreenError("What should I type?")
        if len(value) > MAX_TYPED:
            raise ScreenError("That text is too long.")
        b.set_attr(handle, "AXValue", value)
        done = f"Typed into {name}."
    elif action == "show_menu":
        b.perform(handle, "AXShowMenu")
        done = f"Opened the menu for {name}."
    else:
        raise ScreenError(f"I can't {action} an element.")
    logger.info("ACT %s %s on %s%s", action, id, role,
                f" len={len(value)}" if action == "set_value" else "")
    return done


def line(e: dict, indent: bool = True) -> str:
    pad = "  " * min(e["depth"], 6) if indent else ""
    s = f'{pad}{e["id"]} {e["role"]}'
    if e["label"]:
        s += f' "{e["label"]}"'
    if e["value"]:
        s += f' = "{e["value"]}"'
    return s + ("" if e["enabled"] else " (disabled)")


def outline(snap: dict, max_lines: int = 60) -> str:
    els = snap["elements"][1:]  # e0 is the header
    head = f'{snap["app"]} — {len(snap["elements"])} elements (ids valid until the next look):'
    # 0 interactive, 1 context, 2 rest (static text goes first when over budget)
    tier = lambda e: 0 if e["interactive"] else 1 if e["role"] in {r[2:].lower() for r in CONTEXT} else 2
    budget = max_lines
    if len(els) > budget:
        budget -= 1  # room for the "more not shown" line
    chosen, size = set(), len(head)
    for t in (0, 1, 2):
        for i, e in enumerate(els):
            if tier(e) == t and len(chosen) < budget and size + len(line(e)) < 2900:
                chosen.add(i)
                size += len(line(e)) + 1
    lines = [head] + [line(e) for i, e in enumerate(els) if i in chosen]
    if len(chosen) < len(els):
        lines.append(f"… {len(els) - len(chosen)} more not shown")
    return "\n".join(lines)


def find(query: str, limit: int = 8) -> list[dict]:
    snap = _last
    if snap is None:
        raise ScreenError("Look at the screen first.")
    q = query.lower().strip()
    toks = q.split()
    scored = []
    for e in snap["elements"][1:]:
        label, value, role = e["label"].lower(), (e["value"] or "").lower(), e["role"]
        score = sum(3 * (t in label) + 2 * (t in value) + (t in role) for t in toks)
        if score:
            scored.append((score, difflib.SequenceMatcher(None, q, label).ratio(), e))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [e for _, _, e in scored[:limit]]
