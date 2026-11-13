# 08 — Local web UI protocol

The JARVIS process serves a browser UI on **http://127.0.0.1:8765** (loopback only, same
local-only policy as the models). One process owns the camera; the UI shows exactly what
JARVIS sees.

## HTTP

| Path | Returns |
|---|---|
| `GET /` | `app/ui/static/index.html` |
| `GET /static/<file>` | files under `app/ui/static/` |
| `GET /video.mjpg` | `multipart/x-mixed-replace; boundary=frame` MJPEG of the live camera, ~10 fps. 503 when the camera is off. |

## WebSocket `GET /ws` — JSON text messages

### Server → client

```jsonc
{"type": "hello", "session_id": "…", "model": "gemma4:e2b-it-qat",
 "camera": true, "vision": true,
 "history": [{"role": "user"|"assistant", "text": "…", "ts": 1731570000.0}]}  // recent messages, oldest first

{"type": "status", "state": "idle" | "listening" | "thinking" | "speaking"}

{"type": "user", "id": "…", "text": "…", "ts": 1731570000.0,
 "source": "voice" | "text", "image": "data:image/jpeg;base64,…" | null}  // image = what was attached to this turn

{"type": "assistant_delta", "turn_id": "…", "text": "partial chunk"}       // streamed tokens, append in order
{"type": "assistant", "turn_id": "…", "text": "full reply", "ts": 1731570000.0,
 "latency_ms": 840.0 | null, "outcome": "completed" | "cancelled" | "failed"}

{"type": "vision", "text": "a person at a desk holding a mug", "ts": 1731570000.0}  // ambient camera observation

{"type": "entities", "ts": 1731570000.0, "fps": 4.8, "ms": 33.0, "summary": "person×1, mug×1 (mug entered 3s ago)",
 "items": [{"id": 7, "label": "mug", "conf": 0.82, "box": [x1, y1, x2, y2]}],   // box: normalized xyxy, 0..1
 "events": ["mug entered"]}   // detector frame, at most ~5 Hz; events are debounced enter/leave since the last message

{"type": "log", "ts": 1731570000.12, "level": "INFO", "channel": "stt" | "llm" | "tts" | "vision" | "tool" | "system",
 "logger": "app.audio.asr", "msg": "…"}   // forwarded Python log record

{"type": "tool", "turn_id": "…", "name": "open_app", "args": {"name": "chrome"},
 "ok": true, "result": "opened Google Chrome", "ts": 1731570000.0}   // one tool invocation and its outcome

{"type": "screen", "app": "Google Chrome", "pid": 412, "ts": 1731570000.0, "truncated": false,
 "elements": [{"id": "e12", "parent": "e3" | null, "depth": 2, "role": "button", "label": "Send",
               "value": null, "frame": [x, y, w, h] | null, "enabled": true, "focused": false,
               "actions": ["press"], "interactive": true, "secure": false}]}   // latest accessibility snapshot

{"type": "screen_result", "id": "e12", "action": "press", "ok": true, "result": "Pressed Send.", "ts": 1731570000.0}

{"type": "error", "message": "…"}
```

`log` channel comes from the logger name: `app.audio.*`→stt, `app.reasoning.*`→llm, `app.tts.*`→tts,
`app.perception.*`→vision, `app.tools.*`→tool, anything else under `app.*` or `jarvis.*`→system. The UI's
own and aiohttp loggers are never forwarded, and at most 200 log messages/second are sent (excess dropped).

`turn_id` of `assistant_delta`/`assistant` equals the `id` of the `user` message that started the turn.
A `user` message with no later `assistant` (attention said no) simply gets no reply.

### Client → server

```jsonc
{"type": "text", "text": "what am I holding?", "image": "<base64 JPEG, no data: prefix>" | null,
 "use_camera": false}   // use_camera=true attaches the current camera frame
{"type": "stop"}        // cancel the current turn / playback
{"type": "screen_snapshot", "app": "Google Chrome" | null}                // null/blank = frontmost app
{"type": "screen_action", "id": "e12", "action": "press" | "focus" | "set_value" | "show_menu",
 "value": "text"}       // value only for set_value, max 2000 chars
```

Typed messages are always addressed to JARVIS (they bypass the wake-word attention check).

## Screen (accessibility tree)

JARVIS can read a macOS app's UI as an accessibility tree (voice: "look at Google Chrome"; tools
`look_at_screen`, `find_element`, `press_element`, `type_into`). The SCREEN pane shows the latest snapshot
live: every `screen_snapshot` request or tool-driven snapshot is pushed to all clients as a `screen`
message, and the last one is replayed on connect. These messages work even before the turn manager exists.

- Element ids (`e12`) are only valid until the next snapshot. After a successful `screen_action` the server
  re-snapshots the same app, so the UI refreshes its tree (and drops ids that vanished).
- `screen_action` always answers with a `screen_result` (`ok:false` and a user-facing message on failure or
  after an 8 s timeout); malformed requests get an `error`. The typed value is never logged.
- Secure fields (passwords) are redacted in snapshots and never acted on.
- Requires the Accessibility permission for the terminal that runs JARVIS.

## Vision

When a turn has an image (uploaded, `use_camera`, or a spoken/typed vision request such as
"what do you see") the JPEG goes straight to Gemma 4 in the Ollama chat request
(`"images": [b64]` on the user message). No separate describe-then-reason call.

## Detector

`[detector]` (YOLOE + ByteTrack, optional `vision` extra) runs at `fps` on its own thread and sends
`entities` messages; the camera pane draws the boxes on a canvas over the MJPEG image, and the metrics
strip shows detector fps / latency. Entered/left events are also logged as `app.perception.scene`
(VISION log). The detector gates the VLM (`SceneWatcher` in `app/perception/scene.py`): a call needs a stable
new class, a scene change (track-set Jaccard >= 0.4 or sustained frame change), a stable count change, or the
slow refresh (60 s with a person, 180 s without), with a 3 s cooldown, at most 10 calls/min, and never during a
turn. VLM descriptions still arrive as `vision` messages. Every turn's context also gets a one-line
`[detector]` summary (omitted if the detector is silent for 10 s). Without ultralytics the old timed VLM
cadence is used and no `entities` messages are sent.
