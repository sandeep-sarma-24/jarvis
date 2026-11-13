"""Local web UI: static files, MJPEG camera preview and a WebSocket hub.

Protocol: docs/architecture/08-ui-protocol.md. Implements TurnObserver, so TurnManager
reports progress here without knowing about the web layer. Loopback only.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from app.config import is_loopback
from app.events.types import Event, VisualEvent
from app.orchestration.turns import TurnObserver
from app.perception.jpeg import frame_to_jpeg

logger = logging.getLogger("jarvis.ui")

STATIC_DIR = Path(__file__).parent / "static"
PREVIEW_INTERVAL = 0.1     # ~10 fps
CLIENT_QUEUE_SIZE = 256    # per-websocket backlog before a stalled client is dropped
HISTORY_LIMIT = 20
MAX_WS_MESSAGE = 16 * 1024 * 1024  # an uploaded photo travels as base64 text


@web.middleware
async def _local_only(request: web.Request, handler):
    """Refuse DNS-rebinding (foreign Host) and cross-site pages (foreign Origin)."""
    origin = request.headers.get("Origin")
    if not is_loopback(f"//{request.host}") or (origin and urlsplit(origin).netloc != request.host):
        raise web.HTTPForbidden()
    return await handler(request)


class UiServer(TurnObserver):
    def __init__(self, *, host: str, port: int, camera=None, memory=None, model_name: str = "",
                 vision: bool = False, session_id: str = "") -> None:
        self.turns = None  # set by the owner once TurnManager exists
        self._host, self._port = host, port
        self._camera = camera
        self._memory = memory
        self._model_name = model_name
        self._vision = vision
        self._session_id = session_id
        self._state = "idle"
        self._clients: set[asyncio.Queue] = set()
        self._watchers = 0
        self._jpeg: bytes | None = None
        self._frame = asyncio.Condition()
        self._preview: asyncio.Task | None = None
        self._runner: web.AppRunner | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        app = web.Application(middlewares=[_local_only])
        app.router.add_get("/", self._index)
        app.router.add_get("/video.mjpg", self._video)
        app.router.add_get("/ws", self._ws)
        if STATIC_DIR.is_dir():
            app.router.add_static("/static", STATIC_DIR)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        await web.TCPSite(self._runner, self._host, self._port).start()
        logger.info("Web UI at http://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._preview is not None:
            self._preview.cancel()
        for q in list(self._clients):
            self._close_client(q)
        if self._runner is not None:
            await self._runner.cleanup()

    # ── TurnObserver ───────────────────────────────────────────────────

    def user(self, turn_id, text, source, image_b64) -> None:
        self._broadcast({"type": "user", "id": turn_id, "text": text, "ts": time.time(), "source": source,
                         "image": f"data:image/jpeg;base64,{image_b64}" if image_b64 else None})

    def status(self, state) -> None:
        self._state = state
        self._broadcast({"type": "status", "state": state})

    def delta(self, turn_id, text) -> None:
        self._broadcast({"type": "assistant_delta", "turn_id": turn_id, "text": text})

    def assistant(self, turn_id, text, outcome, latency_ms) -> None:
        self._broadcast({"type": "assistant", "turn_id": turn_id, "text": text, "ts": time.time(),
                         "latency_ms": latency_ms, "outcome": outcome})

    async def on_visual(self, event: Event) -> None:
        if isinstance(event, VisualEvent):
            self._broadcast({"type": "vision", "text": event.description, "ts": event.timestamp})

    def _broadcast(self, msg: dict) -> None:
        for q in list(self._clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:  # stalled client: drop it rather than grow without bound
                logger.warning("UI client too slow; dropped")
                self._close_client(q)

    def _close_client(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)
        while not q.empty():
            q.get_nowait()
        q.put_nowait(None)  # tells the socket writer to close

    # ── HTTP ───────────────────────────────────────────────────────────

    async def _index(self, request: web.Request) -> web.StreamResponse:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(index)

    async def _video(self, request: web.Request) -> web.StreamResponse:
        if self._camera is None:
            raise web.HTTPServiceUnavailable(text="camera is off")
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame", "Cache-Control": "no-store"})
        await resp.prepare(request)
        self._watchers += 1
        if self._preview is None or self._preview.done():
            self._preview = asyncio.create_task(self._preview_loop())
        try:
            sent = None
            while True:
                async with self._frame:
                    await self._frame.wait_for(lambda: self._jpeg is not None and self._jpeg is not sent)
                    sent = self._jpeg
                # Each client awaits its own write, so a slow one only skips frames.
                await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%b\r\n"
                                 % (len(sent), sent))
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._watchers -= 1
            if self._watchers == 0 and self._preview is not None:
                self._preview.cancel()
        return resp

    async def _preview_loop(self) -> None:
        while True:
            t0 = time.monotonic()
            try:
                frame = await self._camera.capture()
                if frame is not None:
                    jpeg = await asyncio.to_thread(frame_to_jpeg, frame)
                    async with self._frame:
                        self._jpeg = jpeg
                        self._frame.notify_all()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Preview frame failed")
            await asyncio.sleep(max(0.0, PREVIEW_INTERVAL - (time.monotonic() - t0)))

    # ── WebSocket ──────────────────────────────────────────────────────

    async def _history(self) -> list[dict]:
        conv = self._memory.conversation if self._memory is not None else None
        if conv is None:
            return []
        try:
            msgs = await asyncio.to_thread(conv.recent_messages, HISTORY_LIMIT)
        except Exception:
            logger.exception("UI history failed")
            return []
        return [{"role": m.role, "text": m.content, "ts": m.timestamp} for m in msgs]

    async def _ws(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(max_msg_size=MAX_WS_MESSAGE)
        await ws.prepare(request)
        q: asyncio.Queue = asyncio.Queue(CLIENT_QUEUE_SIZE)
        q.put_nowait({"type": "hello", "session_id": self._session_id, "model": self._model_name,
                      "camera": self._camera is not None, "vision": self._vision,
                      "history": await self._history()})
        q.put_nowait({"type": "status", "state": self._state})
        self._clients.add(q)

        async def writer() -> None:
            while (msg := await q.get()) is not None:
                await ws.send_json(msg)
            await ws.close()

        wtask = asyncio.create_task(writer())
        try:
            async for raw in ws:
                if raw.type != WSMsgType.TEXT:
                    continue
                try:
                    await self._on_client(json.loads(raw.data))
                except Exception as exc:  # a bad message must not drop the socket or the pipeline
                    logger.warning("UI message rejected: %r", exc)
                    q.put_nowait({"type": "error", "message": str(exc) or type(exc).__name__})
        finally:
            self._clients.discard(q)
            wtask.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await wtask
        return ws

    async def _on_client(self, msg: dict) -> None:
        kind = msg.get("type")
        if self.turns is None:
            raise RuntimeError("JARVIS is still starting")
        if kind == "stop":
            self.turns.stop()
        elif kind == "text":
            text, image = msg.get("text"), msg.get("image")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if image is not None and not isinstance(image, str):
                raise ValueError("image must be a base64 string")
            await self.turns.handle_text(text.strip(), image_b64=image, use_camera=bool(msg.get("use_camera")))
        else:
            raise ValueError(f"unknown message type: {kind!r}")
