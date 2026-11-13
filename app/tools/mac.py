"""macOS tools. Every command is an argv list (no shell); user text never reaches AppleScript source."""
from __future__ import annotations

import difflib
import glob
import re
import subprocess
from pathlib import Path
from typing import Callable
from urllib.parse import quote_plus, urlsplit

from app.tools.registry import Tool, ToolError, ToolRegistry

TIMEOUT = 10.0

# pane -> x-apple.systempreferences id. UNVERIFIED on this macOS build; unknown ids just open Settings.
SETTINGS_PANES = {
    "wifi": "com.apple.Wi-Fi-Settings.extension",
    "bluetooth": "com.apple.BluetoothSettings",
    "network": "com.apple.Network-Settings.extension",
    "notifications": "com.apple.Notifications-Settings.extension",
    "sound": "com.apple.Sound-Settings.extension",
    "focus": "com.apple.Focus-Settings.extension",
    "displays": "com.apple.Displays-Settings.extension",
    "wallpaper": "com.apple.Wallpaper-Settings.extension",
    "battery": "com.apple.Battery-Settings.extension",
    "lock_screen": "com.apple.Lock-Screen-Settings.extension",
    "control_center": "com.apple.ControlCenter-Settings.extension",
    "keyboard": "com.apple.Keyboard-Settings.extension",
    "trackpad": "com.apple.Trackpad-Settings.extension",
    "general": "com.apple.systempreferences.GeneralSettings",
    "software_update": "com.apple.Software-Update-Settings.extension",
    "storage": "com.apple.settings.Storage",
    "accessibility": "com.apple.Accessibility-Settings.extension",
    "privacy": "com.apple.settings.PrivacySecurity.extension",
}

APP_ALIASES = {
    "chrome": "Google Chrome", "google chrome": "Google Chrome",
    "vs code": "Visual Studio Code", "vscode": "Visual Studio Code", "code": "Visual Studio Code",
    "visual studio": "Visual Studio Code",
    "settings": "System Settings", "system settings": "System Settings",
    "preferences": "System Settings", "system preferences": "System Settings",
    "terminal": "Terminal", "command line": "Terminal", "iterm": "iTerm",
    "messages": "Messages", "imessage": "Messages", "texts": "Messages",
    "mail": "Mail", "email": "Mail", "music": "Music", "apple music": "Music",
    "finder": "Finder", "files": "Finder", "activity monitor": "Activity Monitor",
    "task manager": "Activity Monitor", "text edit": "TextEdit", "zoom": "zoom.us",
    "word": "Microsoft Word", "excel": "Microsoft Excel", "powerpoint": "Microsoft PowerPoint",
}

FOLDERS = {"downloads": "~/Downloads", "documents": "~/Documents", "desktop": "~/Desktop", "home": "~"}
_RISKY_SUFFIXES = {".command", ".sh", ".app", ".pkg", ".dmg", ".terminal", ".workflow", ".scpt", ".jar", ".tool"}


def run(argv: list[str]) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        raise ToolError("That took too long. A permission dialog may be waiting.") from None
    except OSError as exc:
        raise ToolError(f"I couldn't run {argv[0]}.") from exc
    if r.returncode != 0:
        err = r.stderr
        if "-1743" in err:
            raise ToolError("I need Automation permission. Enable it in Privacy and Security.")
        raise ToolError("That didn't work.")
    return r.stdout.strip()


def osa(script: str, *args: str) -> str:
    return run(["osascript", "-e", script, *args])


# ── App resolution ────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def load_app_index() -> dict[str, str]:
    """{app name: path} from Spotlight plus the standard folders. Blocking."""
    paths: list[str] = []
    try:
        out = subprocess.run(["mdfind", "kMDItemContentType == 'com.apple.application-bundle'"],
                             capture_output=True, text=True, timeout=TIMEOUT, check=False).stdout
        paths += out.splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    home = str(Path.home())
    for pat in ("/Applications/*.app", "/Applications/*/*.app", "/System/Applications/*.app",
                "/System/Applications/Utilities/*.app", "/System/Library/CoreServices/Finder.app",
                f"{home}/Applications/*.app"):
        paths += glob.glob(pat)
    index: dict[str, str] = {}
    for p in sorted(set(paths), key=lambda p: (not p.startswith("/Applications"), len(p))):
        if not p.endswith(".app") or "/Contents/" in p:
            continue
        if "/Library/" in p and p != "/System/Library/CoreServices/Finder.app":
            continue
        index.setdefault(Path(p).stem, p)
    return index


def resolve_app(spoken: str, index: dict[str, str]) -> tuple[str, str]:
    """Spoken name -> (app name, path). Raises ToolError when unknown or ambiguous."""
    n = _norm(spoken)
    if not n:
        raise ToolError("Which app?")
    by_key = {_norm(name).replace(" ", ""): name for name in index}
    for cand in (APP_ALIASES.get(n), n):
        if cand and (name := by_key.get(_norm(cand).replace(" ", ""))):
            return name, index[name]
    key = n.replace(" ", "")
    tokens = n.split()
    hits = [nm for k, nm in by_key.items() if k.startswith(key)]
    hits = hits or [nm for nm in index if set(tokens) <= set(_norm(nm).split())]
    if len(hits) > 1:
        a, b = sorted(hits, key=len)[:2]
        raise ToolError(f"Did you mean {a} or {b}?")
    if hits:
        return hits[0], index[hits[0]]
    close = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.75)
    if close:
        return by_key[close[0]], index[by_key[close[0]]]
    raise ToolError(f"I couldn't find an app called {spoken}.")


# ── Tool bodies ───────────────────────────────────────────────────────

def _running(app: str) -> bool:  # `is running` does not launch the app
    return osa(f'application "{app}" is running') == "true"  # app comes from a fixed enum


def build_registry(index: Callable[[], dict[str, str]] | dict[str, str] | None = None,
                   extra: list[Tool] = ()) -> ToolRegistry:
    """`index` is a dict or a zero-arg loader (default: real Spotlight scan, built on first use)."""
    cache: dict[str, str] = index if isinstance(index, dict) else {}

    def apps() -> dict[str, str]:
        if not cache:
            cache.update((index or load_app_index)())
        return cache

    def open_app(name: str) -> str:
        app, path = resolve_app(name, apps())
        run(["open", path])
        return f"Opening {app}."

    def open_url(url: str = "", query: str = "") -> str:
        if bool(url) == bool(query):
            raise ToolError("Give either a url or a search query.")
        if query:
            url = "https://www.google.com/search?q=" + quote_plus(query)
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ToolError("I can only open http or https links.")
        run(["open", url])
        return f"Searching for {query}." if query else f"Opening {parts.netloc}."

    def open_path(path: str) -> str:
        key = path.strip().lower().removesuffix(" folder").removeprefix("my ").removeprefix("the ")
        home = Path.home().resolve()
        p = Path(FOLDERS.get(key, path)).expanduser().resolve()
        if not p.exists():
            raise ToolError(f"I couldn't find {path}.")
        if p != home and not p.is_relative_to(home):
            raise ToolError("I only open things in your home folder.")
        if p.suffix.lower() in _RISKY_SUFFIXES:
            raise ToolError("I won't open that kind of file.")
        run(["open", str(p)])
        return f"Opening {p.name or 'home'}."

    def open_settings(pane: str) -> str:
        run(["open", "x-apple.systempreferences:" + SETTINGS_PANES[pane]])
        return f"Opening {pane.replace('_', ' ')} settings."

    def set_volume(level: int | None = None, mute: bool | None = None, delta: int | None = None) -> str:
        if level is None and mute is None and delta is None:
            raise ToolError("Give a level, mute, or delta.")
        if mute is not None:
            osa(f"set volume output muted {'true' if mute else 'false'}")
            if level is None and delta is None:
                return "Muted." if mute else "Unmuted."
        if delta is not None:
            level = max(0, min(100, int(osa("output volume of (get volume settings)")) + delta))
        osa(f"set volume output volume {int(level)}")  # int() keeps the f-string safe
        return f"Volume {level} percent."

    def media_control(action: str, app: str = "") -> str:
        if not app:
            app = next((a for a in ("Spotify", "Music") if _running(a)), "")
        elif not _running(app):
            app = ""
        if not app:
            raise ToolError("No music app is running.")
        verb = {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}[action]
        osa(f'tell application "{app}" to {verb}')
        return {"play": "Playing.", "pause": "Paused."}.get(action, "Skipped." if action == "next" else "Previous track.")

    def list_running_apps() -> str:
        out = osa('tell application "System Events" to get name of (processes where background only is false)')
        return out or "No apps are running."

    def find_files(name: str) -> str:
        out = run(["mdfind", "-onlyin", str(Path.home()), "-name", name])
        found = [p for p in out.splitlines() if "/Library/" not in p and "/." not in p][:10]
        return "\n".join(found) if found else f"No files named {name}."

    S = {"type": "string"}
    return ToolRegistry([
        Tool("open_app", "Open or switch to a Mac app by name, e.g. Safari, Chrome, VS Code.",
             {"type": "object", "properties": {"name": S}, "required": ["name"]}, open_app),
        Tool("open_url", "Open a web link (http/https), or Google-search a query. Give url or query.",
             {"type": "object", "properties": {"url": S, "query": S}}, open_url),
        Tool("open_path", "Open a folder or file: downloads, documents, desktop, home, or a path.",
             {"type": "object", "properties": {"path": S}, "required": ["path"]}, open_path),
        Tool("open_settings", "Open a System Settings pane.",
             {"type": "object", "properties": {"pane": {"type": "string", "enum": list(SETTINGS_PANES)}},
              "required": ["pane"]}, open_settings),
        Tool("set_volume", "Set output volume 0-100, mute or unmute, or change by delta.",
             {"type": "object", "properties": {
                 "level": {"type": "integer", "minimum": 0, "maximum": 100},
                 "mute": {"type": "boolean"},
                 "delta": {"type": "integer", "minimum": -100, "maximum": 100}}}, set_volume),
        Tool("media_control", "Play, pause, skip or go back in Music or Spotify.",
             {"type": "object", "properties": {
                 "action": {"type": "string", "enum": ["play", "pause", "next", "previous"]},
                 "app": {"type": "string", "enum": ["Music", "Spotify"]}}, "required": ["action"]}, media_control),
        Tool("list_running_apps", "List the apps currently open.",
             {"type": "object", "properties": {}}, list_running_apps, info=True),
        Tool("find_files", "Find files in the home folder by name.",
             {"type": "object", "properties": {"name": S}, "required": ["name"]}, find_files, info=True),
        *extra,
    ])
