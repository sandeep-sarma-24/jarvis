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

{"type": "error", "message": "…"}
```

`turn_id` of `assistant_delta`/`assistant` equals the `id` of the `user` message that started the turn.
A `user` message with no later `assistant` (attention said no) simply gets no reply.

### Client → server

```jsonc
{"type": "text", "text": "what am I holding?", "image": "<base64 JPEG, no data: prefix>" | null,
 "use_camera": false}   // use_camera=true attaches the current camera frame
{"type": "stop"}        // cancel the current turn / playback
```

Typed messages are always addressed to JARVIS (they bypass the wake-word attention check).

## Vision

When a turn has an image (uploaded, `use_camera`, or a spoken/typed vision request such as
"what do you see") the JPEG goes straight to Gemma 4 in the Ollama chat request
(`"images": [b64]` on the user message). No separate describe-then-reason call.
