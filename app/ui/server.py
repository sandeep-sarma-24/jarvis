"""Local web UI: static files, MJPEG camera preview and a WebSocket hub.

Protocol: docs/architecture/08-ui-protocol.md. Implements TurnObserver, so TurnManager
reports progress here without knowing about the web layer. Loopback only.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from app import screen
from app.config import is_loopback
from app.events.types import Event, VisualEvent
from app.orchestration.turns import TurnObserver
from app.perception.jpeg import frame_to_jpeg
from app.tools.registry import ToolError

logger = logging.getLogger("jarvis.ui")
screen_log = logging.getLogger("app.tools.screen")  # maps to the "tool" log channel
SCREEN_TIMEOUT = 8.0
SCREEN_ACTIONS = ("press", "focus", "set_value", "show_menu")
_ELEMENT_ID = re.compile(r"e\d+")
MAX_VALUE = 2000

STATIC_DIR = Path(__file__).parent / "static"
PREVIEW_INTERVAL = 0.1     # ~10 fps
CLIENT_QUEUE_SIZE = 256    # per-websocket backlog before a stalled client is dropped
HISTORY_LIMIT = 20
ENTITIES_INTERVAL = 0.19   # entities broadcast cap, ~5 Hz
MAX_WS_MESSAGE = 16 * 1024 * 1024  # an uploaded photo travels as base64 text


_CHANNELS = (("app.audio", "stt"), ("app.reasoning", "llm"), ("app.tts", "tts"),
             ("app.perception", "vision"), ("app.tools", "tool"))
_LOG_SKIP = ("jarvis.ui", "aiohttp")  # the UI's own and transport logs would feed back into the stream
LOG_RATE = 200                        # max log messages forwarded per second; the rest are dropped


def log_channel(name: str) -> str:
    return next((ch for prefix, ch in _CHANNELS if name == prefix or name.startswith(prefix + ".")), "system")


class _LogForwarder(logging.Handler):
    """Forwards log records to WS clients. emit() runs on any thread; delivery hops to the loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, sink) -> None:
        super().__init__()
        self._loop, self._sink = loop, sink
        self._window, self._count = 0, 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_LOG_SKIP):
            return
        msg = {"type": "log", "ts": record.created, "level": record.levelname, "channel": log_channel(record.name),
               "logger": record.name, "msg": record.getMessage()}
        try:
            self._loop.call_soon_threadsafe(self._deliver, msg)
        except RuntimeError:  # loop closed: drop
            pass

    def _deliver(self, msg: dict) -> None:
        now = int(time.monotonic())
        if now != self._window:
            self._window, self._count = now, 0
        self._count += 1
        if self._count <= LOG_RATE:
            self._sink(msg)


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
        self._log_handler: _LogForwarder | None = None
        self._unsub = None
        self._ents_at = 0.0

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
        self._log_handler = _LogForwarder(asyncio.get_running_loop(), self._broadcast)
        for name in ("app", "jarvis"):
            logging.getLogger(name).addHandler(self._log_handler)
        self._wire_screen(asyncio.get_running_loop())
        logger.info("Web UI at http://%s:%d", self._host, self._port)

    def _wire_screen(self, loop: asyncio.AbstractEventLoop) -> None:
        def push(snap: dict) -> None:  # may run on a tool worker thread
            try:
                loop.call_soon_threadsafe(self._broadcast, {"type": "screen", **snap})
            except RuntimeError:  # loop closed
                pass
        self._unsub = screen.subscribe(push)

    async def stop(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        if self._log_handler is not None:
            for name in ("app", "jarvis"):
                logging.getLogger(name).removeHandler(self._log_handler)
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

    def tool(self, turn_id, name, args, ok, result) -> None:
        self._broadcast({"type": "tool", "turn_id": turn_id, "name": name, "args": args, "ok": ok,
                         "result": result, "ts": time.time()})

    async def on_visual(self, event: Event) -> None:
        if isinstance(event, VisualEvent):
            self._broadcast({"type": "vision", "text": event.description, "ts": event.timestamp})

    def entities(self, payload: dict) -> None:
        """Detector update from SceneWatcher; capped at 5 Hz unless it carries enter/leave events."""
        now = time.monotonic()
        if not payload["events"] and now - self._ents_at < ENTITIES_INTERVAL:
            return
        self._ents_at = now
        self._broadcast({"type": "entities", "ts": time.time(), **payload})

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
        if (snap := screen.last()) is not None:
            q.put_nowait({"type": "screen", **snap})
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
        if kind == "screen_snapshot":
            app = msg.get("app")
            if app is not None and not isinstance(app, str):
                raise ValueError("app must be a string or null")
            try:
                await asyncio.wait_for(asyncio.to_thread(screen.snapshot, (app or "").strip() or None), SCREEN_TIMEOUT)
            except TimeoutError:
                raise ValueError("That took too long.") from None
            return
        if kind == "screen_action":
            await self._screen_action(msg)
            return
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

    async def _screen_action(self, msg: dict) -> None:
        eid, action, value = msg.get("id"), msg.get("action"), msg.get("value")
        if not isinstance(eid, str) or not _ELEMENT_ID.fullmatch(eid):
            raise ValueError("id must look like e12")
        if action not in SCREEN_ACTIONS:
            raise ValueError(f"action must be one of {', '.join(SCREEN_ACTIONS)}")
        if action == "set_value":
            if not isinstance(value, str) or len(value) > MAX_VALUE:
                raise ValueError(f"value must be a string of at most {MAX_VALUE} characters")
        elif value is not None:
            raise ValueError("value is only valid for set_value")
        screen_log.info("UI %s %s", action, eid)  # never the value
        last = screen.last()
        try:
            result = await asyncio.wait_for(asyncio.to_thread(screen.act, eid, action, value), SCREEN_TIMEOUT)
            ok = True
        except ToolError as exc:
            ok, result = False, str(exc)
        except TimeoutError:
            ok, result = False, "That took too long."
        self._broadcast({"type": "screen_result", "id": eid, "action": action, "ok": ok, "result": result,
                         "ts": time.time()})
        if ok:  # ids are only valid until the next snapshot, so refresh the tree
            with contextlib.suppress(ToolError, TimeoutError):
                await asyncio.wait_for(asyncio.to_thread(screen.snapshot, last["app"] if last else None),
                                       SCREEN_TIMEOUT)
