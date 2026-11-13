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
