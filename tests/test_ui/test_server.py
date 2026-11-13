"""UiServer over a real loopback socket with a fake camera and fake TurnManager."""
from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.config import ConfigError, JarvisConfig, enforce_local_only
from app.perception.camera import FakeCamera
from app.ui.server import UiServer


class FakeTurns:
    def __init__(self):
        self.texts = []
        self.stopped = 0

    async def handle_text(self, text, image_b64=None, use_camera=False):
        self.texts.append((text, image_b64, use_camera))

    def stop(self):
        self.stopped += 1


async def client_for(ui: UiServer) -> TestClient:
    from aiohttp import web
    from app.ui.server import _local_only

    app = web.Application(middlewares=[_local_only])
    app.router.add_get("/video.mjpg", ui._video)
    app.router.add_get("/ws", ui._ws)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.fixture
async def ui_client():
    made = []

    async def make(camera=None):
        ui = UiServer(host="127.0.0.1", port=0, camera=camera, model_name="m", vision=True, session_id="s1")
        ui.turns = FakeTurns()
        client = await client_for(ui)
        made.append((ui, client))
        return ui, client

    yield make
    for ui, client in made:
        await ui.stop()
        await client.close()


async def test_hello_then_text_reaches_turn_manager_and_broadcasts(ui_client):
    ui, client = await ui_client()
    async with client.ws_connect("/ws") as ws:
        hello = await ws.receive_json()
        assert hello == {"type": "hello", "session_id": "s1", "model": "m", "camera": False,
                         "vision": True, "history": []}
        assert (await ws.receive_json()) == {"type": "status", "state": "idle"}

        await ws.send_json({"type": "text", "text": " hi ", "image": "QUJD", "use_camera": True})
        await ws.send_json({"type": "stop"})
        await ws.send_json({"type": "text", "text": ""})
        err = await ws.receive_json()
        assert err["type"] == "error"
        assert ui.turns.texts == [("hi", "QUJD", True)] and ui.turns.stopped == 1

        ui.delta("t1", "Hel")
        ui.assistant("t1", "Hello", "completed", 12.0)
        assert (await ws.receive_json()) == {"type": "assistant_delta", "turn_id": "t1", "text": "Hel"}
        done = await ws.receive_json()
        assert done["type"] == "assistant" and done["latency_ms"] == 12.0 and done["outcome"] == "completed"


async def test_user_message_image_is_a_data_url(ui_client):
    ui, client = await ui_client()
    async with client.ws_connect("/ws") as ws:
        await ws.receive_json(), await ws.receive_json()
        ui.user("t1", "hello", "text", "QUJD")
        msg = await ws.receive_json()
        assert msg["image"] == "data:image/jpeg;base64,QUJD" and msg["source"] == "text"


async def test_mjpeg_is_503_without_camera(ui_client):
    _, client = await ui_client()
    assert (await client.get("/video.mjpg")).status == 503


async def test_mjpeg_streams_jpeg_parts_from_camera(ui_client):
    cam = FakeCamera()
    await cam.start()
    ui, client = await ui_client(cam)
    resp = await client.get("/video.mjpg")
    assert resp.status == 200
    assert resp.headers["Content-Type"] == "multipart/x-mixed-replace; boundary=frame"
    head = await resp.content.readuntil(b"\r\n\r\n")
    assert head.startswith(b"--frame") and b"image/jpeg" in head
    assert (await resp.content.readexactly(2)) == b"\xff\xd8"
    resp.close()


async def test_foreign_origin_is_refused(ui_client):
    _, client = await ui_client()
    resp = await client.get("/video.mjpg", headers={"Origin": "http://evil.example"})
    assert resp.status == 403


def test_non_loopback_ui_host_is_refused_under_local_only():
    cfg = JarvisConfig()
    cfg.ui.host = "0.0.0.0"
    with pytest.raises(ConfigError):
        enforce_local_only(cfg)
    cfg.runtime.local_only = False
    enforce_local_only(cfg)


async def test_log_forwarder_maps_channels_and_is_thread_safe():
    import asyncio
    import logging
    import threading

    from app.ui.server import _LogForwarder, log_channel

    assert [log_channel(n) for n in ("app.audio.asr", "app.reasoning.llm", "app.tts.x", "app.perception.v",
                                     "app.tools.t", "app.main", "app.audiofoo")] == \
        ["stt", "llm", "tts", "vision", "tool", "system", "system"]

    got = []
    handler = _LogForwarder(asyncio.get_running_loop(), got.append)
    log = logging.getLogger("app.audio.asr")
    log.addHandler(handler)
    try:
        t = threading.Thread(target=lambda: (log.warning("from thread %d", 1),
                                             logging.getLogger("jarvis.ui").warning("skipped")))
        t.start()
        t.join()
        await asyncio.sleep(0.05)
    finally:
        log.removeHandler(handler)
    assert [(m["channel"], m["level"], m["msg"]) for m in got] == [("stt", "WARNING", "from thread 1")]


# ── Screen ──────────────────────────────────────────────────────────────

SNAP = {"app": "Chrome", "pid": 1, "ts": 1.0, "truncated": False, "elements": []}


@pytest.fixture
def fake_screen(monkeypatch):
    from app.ui import server as srv

    subs, calls, state = [], [], {"last": None}

    def snapshot(app=None):
        import threading
        calls.append(("snapshot", app, threading.current_thread() is not threading.main_thread()))
        state["last"] = {**SNAP, "app": app or "Chrome"}
        for fn in subs:
            fn(state["last"])
        return state["last"]

    def act(eid, action, value=None):
        calls.append(("act", eid, action, value))
        if eid == "e9":
            raise srv.ToolError("No such element")
        return "Pressed."

    monkeypatch.setattr(srv.screen, "snapshot", snapshot, raising=False)
    monkeypatch.setattr(srv.screen, "act", act, raising=False)
    monkeypatch.setattr(srv.screen, "last", lambda: state["last"], raising=False)
    monkeypatch.setattr(srv.screen, "subscribe", lambda fn: (subs.append(fn), lambda: subs.remove(fn))[1], raising=False)
    return calls, state, subs


async def _open(ui_client):
    import asyncio

    ui, client = await ui_client()
    ui.turns = None  # screen messages must not need TurnManager
    ui._wire_screen(asyncio.get_running_loop())
    ws = await client.ws_connect("/ws")
    await ws.receive_json(), await ws.receive_json()
    return ui, ws


async def test_screen_snapshot_runs_off_loop_and_broadcasts(ui_client, fake_screen):
    calls, _, _ = fake_screen
    ui, ws = await _open(ui_client)
    await ws.send_json({"type": "screen_snapshot", "app": " Safari "})
    msg = await ws.receive_json()
    assert msg["type"] == "screen" and msg["app"] == "Safari"
    assert calls == [("snapshot", "Safari", True)]
    await ws.send_json({"type": "screen_snapshot", "app": 5})
    assert (await ws.receive_json())["type"] == "error"
    await ws.close()


async def test_screen_action_ok_resnapshots_and_failure_reports(ui_client, fake_screen):
    calls, state, _ = fake_screen
    state["last"] = dict(SNAP)
    ui, ws = await _open(ui_client)
    assert (await ws.receive_json())["type"] == "screen"  # replayed on connect
    await ws.send_json({"type": "screen_action", "id": "e12", "action": "set_value", "value": "hi"})
    res = await ws.receive_json()
    assert (res["type"], res["ok"], res["result"], res["id"], res["action"]) == ("screen_result", True, "Pressed.", "e12", "set_value")
    assert (await ws.receive_json())["type"] == "screen"
    assert calls == [("act", "e12", "set_value", "hi"), ("snapshot", "Chrome", True)]

    await ws.send_json({"type": "screen_action", "id": "e9", "action": "press"})
    bad = await ws.receive_json()
    assert bad["type"] == "screen_result" and bad["ok"] is False and bad["result"] == "No such element"
    await ws.close()


@pytest.mark.parametrize("m", [
    {"id": "12", "action": "press"}, {"id": "e1", "action": "delete"}, {"id": "e1", "action": "set_value"},
    {"id": "e1", "action": "press", "value": "x"}, {"id": "e1", "action": "set_value", "value": "x" * 2001}])
async def test_screen_action_validation(ui_client, fake_screen, m):
    calls, _, _ = fake_screen
    _, ws = await _open(ui_client)
    await ws.send_json({"type": "screen_action", **m})
    assert (await ws.receive_json())["type"] == "error"
    assert calls == []
    await ws.close()


async def test_last_snapshot_sent_on_connect(ui_client, fake_screen):
    _, state, _ = fake_screen
    state["last"] = dict(SNAP)
    ui, client = await ui_client()
    async with client.ws_connect("/ws") as ws:
        await ws.receive_json(), await ws.receive_json()
        assert (await ws.receive_json()) == {"type": "screen", **SNAP}


async def test_stop_unsubscribes(ui_client, fake_screen):
    _, _, subs = fake_screen
    ui, ws = await _open(ui_client)
    assert len(subs) == 1
    await ui.stop()
    assert subs == []
    await ws.close()


async def test_entities_broadcast_is_throttled_but_events_pass(ui_client):
    ui, client = await ui_client()
    async with client.ws_connect("/ws") as ws:
        await ws.receive_json(), await ws.receive_json()
        item = {"id": 7, "label": "mug", "conf": 0.82, "box": [0.1, 0.2, 0.3, 0.4]}
        base = {"fps": 5.0, "ms": 30.0, "summary": "mug×1", "items": [item]}
        ui.entities({**base, "events": []})
        ui.entities({**base, "events": []})                      # inside the 5 Hz window: dropped
        ui.entities({**base, "events": ["mug entered"]})         # events always pass
        a, b = await ws.receive_json(), await ws.receive_json()
        assert a["type"] == "entities" and a["items"] == [item] and a["fps"] == 5.0 and a["events"] == []
        assert b["events"] == ["mug entered"] and b["ts"] > 0
