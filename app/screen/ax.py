"""macOS Accessibility (AX) backend. Raw element access only; tree logic lives in core."""
from __future__ import annotations

import AppKit
import ApplicationServices as AS

from app.screen.core import ScreenError
from app.tools.mac import APP_ALIASES, _norm

_DISABLED = "I need Accessibility permission. Enable it in System Settings, Privacy and Security, Accessibility."
_ATTRS = ["AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXValue", "AXPlaceholderValue",
          "AXHelp", "AXPosition", "AXSize", "AXEnabled", "AXFocused"]
_PLAIN = (str, int, float, bool)


def _fail(err: int) -> None:
    if err == AS.kAXErrorAPIDisabled:
        raise ScreenError(_DISABLED)
    if err == AS.kAXErrorCannotComplete:
        raise ScreenError("That app isn't responding.")
    if err == AS.kAXErrorInvalidUIElement:
        raise ScreenError("That element is gone.")
    raise ScreenError(f"Accessibility error {err}.")


def _plain(v):
    """Keep str/number/bool (NSString/NSNumber arrive as subclasses); drop elements, arrays, etc."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, str):
        return str(v)
    if isinstance(v, (int, float)):
        return v
    return None


def _struct(v, kind):
    if v is None or AS.AXValueGetType(v) != kind:
        return None
    ok, s = AS.AXValueGetValue(v, kind, None)
    return s if ok else None


class AXBackend:
    def trusted(self) -> bool:
        return bool(AS.AXIsProcessTrusted())  # non-prompting variant

    def app_ref(self, app: str | None) -> tuple[str, int, object]:
        ws = AppKit.NSWorkspace.sharedWorkspace()
        if app is None:
            found = ws.frontmostApplication()
            if found is None:
                raise ScreenError("I can't tell which app is in front.")
        else:
            running = [a for a in ws.runningApplications()
                       if a.activationPolicy() == AppKit.NSApplicationActivationPolicyRegular and a.localizedName()]
            want = _norm(APP_ALIASES.get(_norm(app), app))
            key = lambda a: _norm(a.localizedName())
            found = (next((a for a in running if key(a) == want), None)
                     or next((a for a in running if key(a).startswith(want)), None)
                     or next((a for a in running if want in key(a)), None))
            if found is None:
                raise ScreenError(f"{app} isn't running.")
        pid = found.processIdentifier()
        root = AS.AXUIElementCreateApplication(pid)
        AS.AXUIElementSetMessagingTimeout(root, 1.0)
        AS.AXUIElementSetMessagingTimeout(AS.AXUIElementCreateSystemWide(), 1.0)
        # Makes Chromium/Electron build their web tree; harmless elsewhere, errors ignored.
        AS.AXUIElementSetAttributeValue(root, "AXManualAccessibility", True)
        return str(found.localizedName()), pid, root

    def attrs(self, el) -> dict:
        err, vals = AS.AXUIElementCopyMultipleAttributeValues(el, _ATTRS, 0, None)
        if err or vals is None:
            _fail(err)
        # Unsupported/missing attributes come back as AXValue error objects, which _plain/_struct drop.
        v = dict(zip(_ATTRS, vals))
        role, sub = _plain(v["AXRole"]), _plain(v["AXSubrole"])
        secure = "AXSecureTextField" in (role, sub)
        pos, size = _struct(v["AXPosition"], AS.kAXValueCGPointType), _struct(v["AXSize"], AS.kAXValueCGSizeType)
        err, acts = AS.AXUIElementCopyActionNames(el, None)
        enabled = _plain(v["AXEnabled"])
        return {
            "role": role, "subrole": sub,
            "title": _plain(v["AXTitle"]), "description": _plain(v["AXDescription"]),
            "value": None if secure else _plain(v["AXValue"]),
            "placeholder": _plain(v["AXPlaceholderValue"]), "help": _plain(v["AXHelp"]),
            "frame": (pos.x, pos.y, size.width, size.height) if pos and size else None,
            "enabled": True if enabled is None else bool(enabled),
            "focused": bool(_plain(v["AXFocused"])),
            "actions": [str(a) for a in acts] if not err and acts else [],
        }

    def children(self, el) -> list:
        # Plain AXChildren everywhere; for an app root that already includes windows and the menu bar.
        err, kids = AS.AXUIElementCopyAttributeValue(el, "AXChildren", None)
        return list(kids) if not err and kids else []

    def perform(self, el, action: str) -> None:
        err = AS.AXUIElementPerformAction(el, action)
        if err:
            _fail(err)

    def set_attr(self, el, attr: str, value) -> None:
        err = AS.AXUIElementSetAttributeValue(el, attr, value)
        if err:
            _fail(err)
