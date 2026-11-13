"""Tool registry, router, app resolution and the turn flow. subprocess is always mocked."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from app.tools import mac
from app.tools.registry import ToolError
from app.tools.router import route
from tests.test_orchestration.fakes import FakeTTS, StreamingModel, make_manager, transcript, wait_for
from tests.test_orchestration.test_turns import finish

INDEX = {
    "Google Chrome": "/Applications/Google Chrome.app", "Safari": "/Applications/Safari.app",
    "Visual Studio Code": "/Applications/Visual Studio Code.app", "System Settings": "/System/Applications/System Settings.app",
    "Spotify": "/Applications/Spotify.app", "Microsoft Word": "/Applications/Microsoft Word.app",
    "Microsoft Excel": "/Applications/Microsoft Excel.app", "zoom.us": "/Applications/zoom.us.app",
}


@pytest.fixture
def calls(monkeypatch):
    log = []

    def fake_run(argv, **kw):
        log.append(argv)
        out = "true" if "is running" in " ".join(argv) else "50"
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(mac.subprocess, "run", fake_run)
    return log


@pytest.fixture
def reg():
    return mac.build_registry(dict(INDEX))


# ── Router ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("Open Chrome.", ("open_app", {"name": "chrome"})),
    ("jarvis, please open vs code", ("open_app", {"name": "vs code"})),
    ("open wifi settings", ("open_settings", {"pane": "wifi"})),
    ("open Wi-Fi settings", ("open_settings", {"pane": "wifi"})),
    ("open system settings", ("open_app", {"name": "system settings"})),
    ("volume to 30", ("set_volume", {"level": 30})),
    ("set the volume to 30 percent", ("set_volume", {"level": 30})),
    ("volume up", ("set_volume", {"delta": 10})),
    ("mute", ("set_volume", {"mute": True})),
    ("unmute the volume", ("set_volume", {"mute": False})),
    ("open downloads", ("open_path", {"path": "downloads"})),
    ("search google for best pizza", ("open_url", {"query": "best pizza"})),
    ("open github.com", ("open_url", {"url": "https://github.com"})),
    ("pause music", ("media_control", {"action": "pause"})),
    ("next song", ("media_control", {"action": "next"})),
    ("look at Google Chrome", ("look_at_screen", {"app": "google chrome"})),
    ("jarvis look at the screen", ("look_at_screen", {})),
    ("look at finder window", ("look_at_screen", {"app": "finder"})),
])
def test_router_hits(text, expected):
    assert route(text) == expected


@pytest.mark.parametrize("text", [
    "what does open mean", "open chrome and mute", "how are you", "what is the volume", "tell me a joke",
    "look at this", "look at me",
])
def test_router_falls_through(text):
    assert route(text) is None


def test_every_route_validates(reg):
    for text in ["open chrome", "open wifi settings", "volume to 30", "mute", "volume down", "open downloads",
                 "search google for cats", "open github.com", "pause music", "previous track"]:
        name, args = route(text)
        reg.validate(name, args)


# ── Validation ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,args", [
    ("set_volume", {"level": 150}), ("set_volume", {"level": -1}), ("set_volume", {"level": "loud"}),
    ("open_settings", {"pane": "secret_pane"}), ("open_settings", {}), ("open_app", {"name": "x", "evil": 1}),
    ("rm_rf", {}), ("media_control", {"action": "explode"}),
])
def test_validate_rejects(reg, name, args):
    with pytest.raises(ToolError):
        reg.validate(name, args)


def test_validate_coerces_int_string(reg):
    assert reg.validate("set_volume", {"level": "30"}) == {"level": 30}


@pytest.mark.parametrize("args", [
    {"url": "file:///etc/passwd"}, {"url": "javascript:alert(1)"}, {"url": "ssh://host"}, {"url": "http://"}, {},
    {"url": "https://a.com", "query": "x"},
])
def test_open_url_rejects_bad(reg, calls, args):
    res = reg.execute("open_url", args)
    assert not res.ok and calls == []


def test_open_url_and_volume_commands(reg, calls):
    assert reg.execute("open_url", {"url": "https://example.com/a"}).ok
    assert calls[-1] == ["open", "https://example.com/a"]
    assert reg.execute("open_url", {"query": "a b&c"}).result == "Searching for a b&c."
    assert calls[-1] == ["open", "https://www.google.com/search?q=a+b%26c"]
    reg.execute("set_volume", {"level": 30})
    assert calls[-1] == ["osascript", "-e", "set volume output volume 30"]
    assert reg.execute("set_volume", {"mute": True}).result == "Muted."
    assert reg.execute("set_volume", {"level": 101}).ok is False


def test_open_path(reg, calls, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "x.command").write_text("")
    assert reg.execute("open_path", {"path": "downloads"}).ok
    assert calls[-1] == ["open", str((tmp_path / "Downloads").resolve())]
    assert not reg.execute("open_path", {"path": "nope"}).ok
    assert not reg.execute("open_path", {"path": "/etc"}).ok
    assert not reg.execute("open_path", {"path": "~/x.command"}).ok


def test_settings_and_media(reg, calls):
    reg.execute("open_settings", {"pane": "wifi"})
    assert calls[-1] == ["open", "x-apple.systempreferences:com.apple.Wi-Fi-Settings.extension"]
    assert reg.execute("media_control", {"action": "pause"}).result == "Paused."
    assert calls[-1][-1] == 'tell application "Spotify" to pause'


def test_subprocess_timeout_is_a_spoken_failure(reg, monkeypatch):
    def boom(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 10)
    monkeypatch.setattr(mac.subprocess, "run", boom)
    res = reg.execute("set_volume", {"level": 5})
    assert not res.ok and "too long" in res.result


# ── App resolution ─────────────────────────────────────────────────────

@pytest.mark.parametrize("spoken,name", [
    ("chrome", "Google Chrome"), ("vs code", "Visual Studio Code"), ("settings", "System Settings"),
    ("safari", "Safari"), ("safarii", "Safari"), ("spot if i", "Spotify"), ("zoom", "zoom.us"),
    ("visual studio code", "Visual Studio Code"),
])
def test_resolve_app(spoken, name):
    assert mac.resolve_app(spoken, INDEX)[0] == name


def test_resolve_app_ambiguous_and_unknown():
    with pytest.raises(ToolError, match="Did you mean"):
        mac.resolve_app("microsoft", INDEX)
    with pytest.raises(ToolError, match="couldn't find"):
        mac.resolve_app("blender", INDEX)


def test_open_app_launches_resolved_path(reg, calls):
    assert reg.execute("open_app", {"name": "chrome"}).result == "Opening Google Chrome."
    assert calls == [["open", "/Applications/Google Chrome.app"]]


# ── Turn flow ──────────────────────────────────────────────────────────

class ToolModel(StreamingModel):
    """First pass emits tool calls; a pass without tools streams the follow-up text."""

    def __init__(self, calls, follow=("Safari and Mail are open. ",)):
        super().__init__([])
        self.tool_calls, self.follow, self.seen = calls, list(follow), []

    def stream(self, messages, *, max_tokens=256, temperature=0.2, cancel=None, tools=None, on_tool_calls=None):
        self.seen.append((messages, tools))
        if tools:
            on_tool_calls([{"function": {"name": n, "arguments": a}} for n, a in self.tool_calls])
            return
        yield from self.follow


class Obs:
    def __init__(self):
        self.tools = []

    def __getattr__(self, name):
        if name == "tool":
            return lambda *a: self.tools.append(a)
        return lambda *a: None


async def _turn(model, text, calls_log, monkeypatch=None):
    obs, tts = Obs(), FakeTTS()
    tm, *_ = make_manager(model=model, tts=tts, observer=obs)
    tm._tools = mac.build_registry(dict(INDEX))
    await tm.handle_transcript(transcript(text))
    await finish(tm)
    return tm, obs, tts


async def test_router_turn_skips_llm(calls):
    model = ToolModel([])
    tm, obs, tts = await _turn(model, "jarvis open chrome", calls)
    assert model.seen == []
    assert calls == [["open", "/Applications/Google Chrome.app"]]
    assert [t[1:] for t in obs.tools] == [("open_app", {"name": "chrome"}, True, "Opening Google Chrome.")]
    assert [t for _, t in tts.calls] == ["Opening Google Chrome."]


async def test_model_tool_call_speaks_template_without_second_pass(calls):
    model = ToolModel([("set_volume", {"level": 30}), ("open_app", {"name": "safari"})])
    tm, obs, tts = await _turn(model, "jarvis make it quieter and get me safari", calls)
    assert len(model.seen) == 1 and model.seen[0][1]  # tools were offered once
    assert calls == [["osascript", "-e", "set volume output volume 30"], ["open", "/Applications/Safari.app"]]
    assert [t for _, t in tts.calls] == ["Volume 30 percent.", "Opening Safari."]


async def test_max_three_calls_and_bad_args_reported(calls):
    model = ToolModel([("set_volume", {"level": 999})] + [("open_app", {"name": "safari"})] * 4)
    tm, obs, tts = await _turn(model, "jarvis do things and stuff", calls)
    assert len(obs.tools) == 3
    assert obs.tools[0][3] is False and obs.tools[1][3] is True


async def test_info_tool_gets_second_pass_without_tools(calls):
    model = ToolModel([("list_running_apps", {})])
    tm, obs, tts = await _turn(model, "jarvis what is open and running right now", calls)
    assert len(model.seen) == 2 and model.seen[1][1] is None
    roles = [m["role"] for m in model.seen[1][0]]
    assert roles[-2:] == ["assistant", "tool"]
    assert [t for _, t in tts.calls] == ["Safari and Mail are open."]


async def test_routed_info_tool_gets_second_pass(calls):
    from app.tools.registry import Tool
    model = ToolModel([], follow=("Chrome shows a search box. ",))
    obs, tts = Obs(), FakeTTS()
    tm, *_ = make_manager(model=model, tts=tts, observer=obs)
    tm._tools = mac.build_registry(dict(INDEX), extra=[Tool(
        "look_at_screen", "", {"type": "object", "properties": {"app": {"type": "string"}}},
        lambda app="": "e1 searchfield \"Search\"", info=True)])
    await tm.handle_transcript(transcript("jarvis look at chrome"))
    await finish(tm)
    assert len(model.seen) == 1 and model.seen[0][1] is None
    assert model.seen[0][0][-1] == {"role": "tool", "tool_name": "look_at_screen", "content": 'e1 searchfield "Search"'}
    assert [t for _, t in tts.calls] == ["Chrome shows a search box."]


async def test_plain_answer_still_streams(calls):
    class Plain(ToolModel):
        def stream(self, messages, **kw):
            yield from ["Paris is the capital. ", "Of France."]
    tm, obs, tts = await _turn(Plain([]), "jarvis capital of france please", calls)
    assert tts.calls[0][1] == "Paris is the capital." and obs.tools == [] and calls == []
