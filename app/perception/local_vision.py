"""Local vision analyzer — Ollama-backed image understanding.

Sends frames to a local Ollama vision model (e.g. gemma4) via /api/chat.
Produces structured VisualAnalysis from the model's text response.

Follows the same sync-httpx + asyncio.to_thread pattern as OllamaModel
in app/reasoning/local_model.py. No new dependencies.

No raw frame bytes leave this module except as base64 to the local model.
No cloud APIs. No raw frame persistence.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Protocol, runtime_checkable

import httpx
import numpy as np
from PIL import Image

from app.perception.types import Detection, Frame, VisualAnalysis, VisualEventType

logger = logging.getLogger(__name__)

_EVENT_TYPE_MAP: dict[str, VisualEventType] = {
    "SCENE_CHANGE": VisualEventType.SCENE_CHANGE,
    "PERSON_DETECTED": VisualEventType.PERSON_DETECTED,
    "OBJECT_DETECTED": VisualEventType.OBJECT_DETECTED,
    "TEXT_DETECTED": VisualEventType.TEXT_DETECTED,
    "VISUAL_SUMMARY": VisualEventType.VISUAL_SUMMARY,
}

_DEFAULT_PROMPT = (
    "Analyze this image and respond with ONLY a JSON object (no markdown, no explanation):\n"
    '{"event_type": "<one of: SCENE_CHANGE, PERSON_DETECTED, OBJECT_DETECTED, '
    'TEXT_DETECTED, VISUAL_SUMMARY>", "description": "<detailed 2-4 sentence description '
    'including people, their actions, posture, clothing, objects, and environment>", '
    '"confidence": <0.0-1.0>, "detections": [{"label": "<object>", "confidence": <0.0-1.0>}]}'
)


@runtime_checkable
class VisionBackend(Protocol):
    """Backend for local vision inference. Decouples from Ollama specifics."""

    def infer(self, image_b64: str, prompt: str) -> str: ...


class OllamaVisionBackend:
    """Vision backend using Ollama's /api/chat with images field.

    ponytail: sync httpx, caller wraps in asyncio.to_thread.
    Upgrade path: async httpx client if thread overhead matters.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
        temperature: float = 0.1,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds)
        self._temperature = temperature
        # Must match reasoning: Ollama reloads a model when num_ctx changes between requests.
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive

    def infer(self, image_b64: str, prompt: str) -> str:
        resp = self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
                "think": False,
                "keep_alive": self._keep_alive,
                "options": {
                    "temperature": self._temperature,
                    "num_predict": 512,
                    "num_ctx": self._num_ctx,
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")

    def close(self) -> None:
        self._client.close()


class LocalVisionAnalyzer:
    """VisualAnalyzer backed by a local vision model.

    Frame -> preprocess (resize, encode JPEG) -> VisionBackend -> parse JSON -> VisualAnalysis.

    Handles timeouts, malformed output, and backend failures without crashing.
    """

    def __init__(
        self,
        *,
        backend: VisionBackend,
        prompt: str = "",
        timeout_seconds: float = 30.0,
        max_image_size: int = 1024,
    ) -> None:
        self._backend = backend
        self._prompt = prompt if prompt else _DEFAULT_PROMPT
        self._timeout = timeout_seconds
        self._max_size = max_image_size

        self.inference_count = 0
        self.inference_success = 0
        self.inference_failure = 0
        self.inference_timeout = 0
        self.parse_failure = 0
        self.backend_failure = 0
        self.preprocessing_latency_ms = 0.0
        self.inference_latency_ms = 0.0

    async def analyze(self, frame: Frame) -> VisualAnalysis | None:
        self.inference_count += 1

        t_pre = time.perf_counter()
        image_b64 = self._preprocess(frame)
        self.preprocessing_latency_ms = (time.perf_counter() - t_pre) * 1000

        if image_b64 is None:
            self.inference_failure += 1
            self.backend_failure += 1
            return None

        t_infer = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(self._backend.infer, image_b64, self._prompt),
                timeout=self._timeout,
            )
        except TimeoutError:
            self.inference_timeout += 1
            self.inference_failure += 1
            logger.warning("Vision inference timed out after %.1fs", self._timeout)
            return None
        except Exception:
            self.backend_failure += 1
            self.inference_failure += 1
            logger.exception("Vision inference failed")
            return None

        self.inference_latency_ms = (time.perf_counter() - t_infer) * 1000

        result = self._parse(raw, frame.timestamp)
        if result is not None:
            self.inference_success += 1
        else:
            self.parse_failure += 1
            self.inference_failure += 1
        return result

    def _preprocess(self, frame: Frame) -> str | None:
        """Resize and JPEG-encode a frame, return base64 string.

        Returns None if encoding fails — caller treats as preprocessing error.
        """
        if frame.format in ("jpeg", "jpg"):
            return base64.b64encode(frame.data).decode()

        channels = 3
        expected = frame.width * frame.height * channels
        if len(frame.data) < expected:
            logger.warning(
                "Frame data too short: got %d bytes, expected %d", len(frame.data), expected,
            )
            return None

        arr = np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(
            frame.height, frame.width, channels,
        )

        if frame.format == "bgr24":
            arr = arr[:, :, ::-1]

        h, w = arr.shape[:2]
        if max(h, w) > self._max_size:
            scale = self._max_size / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            img = Image.fromarray(arr)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            arr = np.asarray(img)

        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    def _parse(self, raw: str, timestamp: float) -> VisualAnalysis | None:
        """Parse model output into VisualAnalysis. Tolerant of messy JSON."""
        text = raw.strip()

        if not text:
            logger.warning("Vision model returned empty output")
            return None

        # Strip <think> blocks but keep any JSON that might be inside them as fallback
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # If stripping think blocks left nothing, look for JSON inside the think block
        if not text and think_match:
            text = think_match.group(1).strip()

        if not text:
            logger.warning("Vision model returned no content outside think block")
            return None

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    logger.warning("Vision model returned unparseable output: %r", text[:200])
                    return None
            else:
                logger.warning("Vision model returned non-JSON output: %r", text[:200])
                return None

        if not isinstance(data, dict):
            logger.warning("Vision model returned non-object JSON")
            return None

        event_type_str = str(data.get("event_type", "VISUAL_SUMMARY")).upper()
        event_type = _EVENT_TYPE_MAP.get(event_type_str, VisualEventType.VISUAL_SUMMARY)

        description = str(data.get("description", ""))
        if not description:
            logger.warning("Vision model returned empty description")
            return None

        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        detections: list[Detection] = []
        raw_dets = data.get("detections", [])
        if isinstance(raw_dets, list):
            for d in raw_dets:
                if isinstance(d, dict) and "label" in d:
                    try:
                        det_conf = float(d.get("confidence", 0.5))
                        det_conf = max(0.0, min(1.0, det_conf))
                    except (TypeError, ValueError):
                        det_conf = 0.5
                    detections.append(Detection(label=str(d["label"]), confidence=det_conf))

        return VisualAnalysis(
            timestamp=timestamp,
            description=description,
            event_type=event_type,
            detections=tuple(detections),
            confidence=confidence,
        )

    def close(self) -> None:
        """Propagate shutdown to the backend."""
        if hasattr(self._backend, "close"):
            self._backend.close()

    @property
    def metrics(self) -> dict[str, int | float]:
        return {
            "vision_inference_count": self.inference_count,
            "vision_inference_success": self.inference_success,
            "vision_inference_failure": self.inference_failure,
            "vision_inference_timeout": self.inference_timeout,
            "vision_parse_failure": self.parse_failure,
            "vision_backend_failure": self.backend_failure,
            "vision_inference_latency_ms": round(self.inference_latency_ms, 1),
            "vision_preprocessing_latency_ms": round(self.preprocessing_latency_ms, 1),
        }
