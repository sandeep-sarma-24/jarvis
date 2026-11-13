import pytest

from app import screen
from app.screen import core
from app.screen.core import NOT_TRUSTED, ScreenError
from app.screen.tools import screen_tools
from app.tools.registry import ToolRegistry


class N:
    def __init__(self, role, title="", kids=(), **kw):
        self.a = {"role": role, "subrole": None, "title": title, "description": "", "value": None,
                  "placeholder": "", "help": "", "frame": (0, 0, 10, 10), "enabled": True,
                  "focused": False,
                  "actions": ["AXPress"] if role in ("AXButton", "AXLink") else []}
        self.a.update(kw)
        self.kids, self.gone, self.boom = list(kids), False, False


class Fake:
    def __init__(self, root_kids):
        self.root, self.ok, self.calls = N("AXApplication", kids=root_kids), True, []

    def trusted(self):
        return self.ok

    def app_ref(self, app):
        return "Chrome", 42, self.root

    def attrs(self, el):
        if el.gone:
            raise ScreenError("stale")
        return dict(el.a)

    def children(self, el):
        if el.boom:
            raise ScreenError("boom")
        return el.kids

    def perform(self, el, action):
        self.calls.append(("perform", el, action))

    def set_attr(self, el, attr, value):
        self.calls.append(("set", el, attr, value))


@pytest.fixture
def setup():
    def make(kids):
        fake = Fake(kids)
        screen.set_backend(fake)
        return fake
    yield make
    screen.set_backend(None)
    core._cache.clear()
    core._subs.clear()
    core._last = None


def roles(snap):
    return [(e["id"], e["role"], e["label"], e["parent"]) for e in snap["elements"]]


def test_pruning_ids_and_reparenting(setup):
    send = N("AXButton", "Send", [N("AXStaticText", "", value="Send")])
    setup([N("AXWindow", "Main", [N("AXGroup", "", [N("AXGroup", "", [send,
           N("AXStaticText", "", value="Hello")])])])])
    snap = screen.snapshot()
    assert roles(snap) == [("e0", "application", "Chrome", None), ("e1", "window", "Main", "e0"),
                           ("e2", "button", "Send", "e1"), ("e3", "statictext", "Hello", "e1")]
    assert snap["elements"][2]["depth"] == 2 and snap["elements"][2]["actions"] == ["press"]
    assert screen.last() is snap and not snap["truncated"]


def test_caps_and_broken_subtree(setup, monkeypatch):
    bad = N("AXGroup", "g", [N("AXButton", "hidden")])
    bad.boom = True
    bad2 = N("AXButton", "stale"); bad2.gone = True
    setup([bad, bad2, N("AXButton", "ok")])
    snap = screen.snapshot()
    assert [e["label"] for e in snap["elements"]] == ["Chrome", "g", "ok"]
    monkeypatch.setattr(core, "MAX_ELEMENTS", 5)
    setup([N("AXButton", f"b{i}") for i in range(10)])
    snap = screen.snapshot()
    assert len(snap["elements"]) == 5 and snap["truncated"]


def test_secure_redacted_and_refused(setup):
    f = setup([N("AXSecureTextField", "Password", value="hunter2"),
               N("AXTextField", "Pw2", subrole="AXSecureTextField", value="x")])
    snap = screen.snapshot()
    assert snap["elements"][1]["value"] == "•••" and snap["elements"][1]["secure"]
    assert snap["elements"][2]["secure"]
    for i in ("e1", "e2"):
        for action, v in (("press", None), ("set_value", "x")):
            with pytest.raises(ScreenError, match="password"):
                screen.act(i, action, v)
    assert f.calls == []


def test_act_happy_paths(setup):
    btn, field = N("AXButton", "Send"), N("AXTextField", "Search")
    f = setup([btn, field])
    screen.snapshot()
    assert screen.act("e1", "press") == "Pressed Send."
    assert screen.act("e2", "set_value", "cats") == "Typed into Search."
    assert screen.act("e2", "focus") == "Focused Search."
    assert f.calls == [("perform", btn, "AXPress"), ("set", field, "AXValue", "cats"),
                       ("set", field, "AXFocused", True)]


def test_act_errors(setup):
    btn, dis = N("AXButton", "Send"), N("AXButton", "Off", enabled=False)
    setup([btn, dis, N("AXLink", "Docs", actions=[])])
    screen.snapshot()
    with pytest.raises(ScreenError, match="don't know element e99"):
        screen.act("e99", "press")
    with pytest.raises(ScreenError, match="Off is disabled"):
        screen.act("e2", "press")
    with pytest.raises(ScreenError, match="isn't a text field"):
        screen.act("e1", "set_value", "x")
    with pytest.raises(ScreenError, match="can't be pressed"):
        screen.act("e3", "press")
    btn.a["role"] = "AXCheckBox"
    with pytest.raises(ScreenError, match="e1 is gone"):
        screen.act("e1", "press")
    btn.gone = True
    with pytest.raises(ScreenError, match="e1 is gone"):
        screen.act("e1", "press")


def test_not_trusted(setup):
    f = setup([N("AXButton", "Send")])
    screen.snapshot()
    f.ok = False
    with pytest.raises(ScreenError) as e:
        screen.snapshot()
    assert str(e.value) == NOT_TRUSTED
    with pytest.raises(ScreenError, match="Accessibility"):
        screen.act("e1", "press")


def test_outline_format_and_priority(setup):
    setup([N("AXWindow", "Main", [N("AXTextField", "Search", value="cats"), N("AXLink", "Docs")])])
    text = screen.outline(screen.snapshot())
    assert text.splitlines() == ['Chrome — 4 elements (ids valid until the next look):',
                                 '  e1 window "Main"', '    e2 textfield "Search" = "cats"', '    e3 link "Docs"']
    setup([N("AXWindow", "Main", [N("AXStaticText", "", value=f"t{i}") for i in range(70)]
             + [N("AXButton", "Go")])])
    lines = screen.outline(screen.snapshot()).splitlines()
    assert any('button "Go"' in x for x in lines) and any("window" in x for x in lines)
    assert len(lines) <= 61 and lines[-1].endswith("more not shown") and "t0" in "".join(lines)
    setup([N("AXButton", f"b{i}") for i in range(100)])
    text = screen.outline(screen.snapshot())
    assert len(text) <= 3000 and text.splitlines()[-1] == "… 41 more not shown"


def test_find_ranking(setup):
    setup([N("AXButton", "Send later"), N("AXButton", "Send"), N("AXTextField", "Search", value="send me")])
    screen.snapshot()
    assert [e["id"] for e in screen.find("send")] == ["e2", "e1", "e3"]  # exact label, longer label, value
    with pytest.raises(ScreenError):
        core._last = None
        screen.find("x")


def test_subscribe(setup):
    setup([N("AXButton", "Send")])
    seen = []
    off = screen.subscribe(seen.append)
    screen.subscribe(lambda s: 1 / 0)  # must not break snapshot
    snap = screen.snapshot()
    assert seen == [snap]
    off()
    screen.snapshot()
    assert len(seen) == 1


def test_tools_end_to_end(setup):
    f = setup([N("AXButton", "Send"), N("AXTextField", "Search")])
    reg = ToolRegistry(screen_tools())
    r = reg.execute("look_at_screen", {})
    assert r.ok and r.info and 'e1 button "Send"' in r.result
    assert reg.execute("find_element", {"query": "search"}).result == 'e2 textfield "Search"'
    assert reg.execute("find_element", {"query": "zzz"}).result == "No element matches zzz."
    assert reg.execute("press_element", {"id": "e1"}).result == "Pressed Send."
    assert reg.execute("type_into", {"id": "e2", "text": "hi"}).result == "Typed into Search."
    bad = reg.execute("press_element", {"id": "e9"})
    assert not bad.ok and "don't know" in bad.result
    assert len(f.calls) == 2
