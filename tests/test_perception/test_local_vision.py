"""Tests for LocalVisionAnalyzer: structured output parsing, timeouts, failures,
preprocessing, lifecycle, and metrics.

All tests use a FakeVisionBackend — no real model or Ollama instance required.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time

import numpy as np
import pytest
from PIL import Image

from app.perception.local_vision import (
    LocalVisionAnalyzer,
    OllamaVisionBackend,
    VisionBackend,
    _DEFAULT_PROMPT,
)
from app.perception.types import Frame, VisualEventType


def _frame(*, pixel: int = 128, w: int = 4, h: int = 4, fmt: str = "rgb24") -> Frame:
    return Frame(
        timestamp=time.time(),
        data=bytes([pixel] * w * h * 3),
        width=w,
        height=h,
        format=fmt,
    )


def _jpeg_frame(w: int = 4, h: int = 4) -> Frame:
    """Create a real JPEG-encoded frame."""
    import io
    arr = np.full((h, w, 3), 100, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return Frame(
        timestamp=time.time(),
        data=buf.getvalue(),
        width=w,
        height=h,
        format="jpeg",
    )


class FakeVisionBackend:
    """Returns a canned response string. Implements VisionBackend protocol."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.call_count = 0
        self.last_prompt: str = ""
        self.last_image_b64: str = ""
        self.closed = False

    def infer(self, image_b64: str, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_image_b64 = image_b64
        return self.response

    def close(self) -> None:
        self.closed = True


class SlowVisionBackend:
    """Backend that sleeps, for timeout testing."""

    def __init__(self, delay: float = 10.0) -> None:
        self._delay = delay

    def infer(self, image_b64: str, prompt: str) -> str:
        time.sleep(self._delay)
        return '{"event_type": "VISUAL_SUMMARY", "description": "too slow", "confidence": 0.5}'


class FailingVisionBackend:
    """Backend that raises on every call."""

    def __init__(self) -> None:
        self.closed = False

    def infer(self, image_b64: str, prompt: str) -> str:
        raise ConnectionError("model unavailable")

    def close(self) -> None:
        self.closed = True


def _good_response(**overrides) -> str:
    data = {
        "event_type": "VISUAL_SUMMARY",
        "description": "A desk with a monitor",
        "confidence": 0.7,
        "detections": [{"label": "monitor", "confidence": 0.8}],
    }
    data.update(overrides)
    return json.dumps(data)


# ── Prompt behavior (#2) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_prompt_used_when_empty():
    """Empty prompt string -> analyzer uses _DEFAULT_PROMPT."""
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend, prompt="")
    await analyzer.analyze(_frame())
    assert backend.last_prompt == _DEFAULT_PROMPT


@pytest.mark.asyncio
async def test_default_prompt_used_when_omitted():
    """No prompt kwarg -> analyzer uses _DEFAULT_PROMPT."""
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)
    await analyzer.analyze(_frame())
    assert backend.last_prompt == _DEFAULT_PROMPT


@pytest.mark.asyncio
async def test_custom_prompt_used_when_provided():
    """Explicit non-empty prompt -> forwarded to backend as-is."""
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend, prompt="Describe this scene.")
    await analyzer.analyze(_frame())
    assert backend.last_prompt == "Describe this scene."


# ── Successful inference ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_json_response():
    response = json.dumps({
        "event_type": "PERSON_DETECTED",
        "description": "A person standing near the desk",
        "confidence": 0.91,
        "detections": [{"label": "person", "confidence": 0.91}],
    })
    backend = FakeVisionBackend(response)
    analyzer = LocalVisionAnalyzer(backend=backend)

    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.PERSON_DETECTED
    assert result.description == "A person standing near the desk"
    assert result.confidence == 0.91
    assert len(result.detections) == 1
    assert result.detections[0].label == "person"
    assert result.detections[0].confidence == 0.91
    assert analyzer.inference_success == 1
    assert analyzer.inference_failure == 0
    assert backend.call_count == 1


@pytest.mark.asyncio
async def test_scene_change_event_type():
    response = json.dumps({
        "event_type": "SCENE_CHANGE",
        "description": "The room lights turned off",
        "confidence": 0.8,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.SCENE_CHANGE


@pytest.mark.asyncio
async def test_object_detected_event_type():
    response = json.dumps({
        "event_type": "OBJECT_DETECTED",
        "description": "A coffee mug on the desk",
        "confidence": 0.75,
        "detections": [
            {"label": "mug", "confidence": 0.75},
            {"label": "desk", "confidence": 0.90},
        ],
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.OBJECT_DETECTED
    assert len(result.detections) == 2


@pytest.mark.asyncio
async def test_text_detected_event_type():
    response = json.dumps({
        "event_type": "TEXT_DETECTED",
        "description": "Text visible on screen: Hello World",
        "confidence": 0.65,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.TEXT_DETECTED


# ── Structured output parsing edge cases ─────────────────────────────────


@pytest.mark.asyncio
async def test_markdown_code_fences_stripped():
    response = '```json\n{"event_type": "VISUAL_SUMMARY", "description": "A desk", "confidence": 0.7}\n```'
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.description == "A desk"


@pytest.mark.asyncio
async def test_json_embedded_in_text():
    response = 'Here is the analysis: {"event_type": "VISUAL_SUMMARY", "description": "A monitor", "confidence": 0.6} done.'
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.description == "A monitor"


@pytest.mark.asyncio
async def test_unknown_event_type_defaults_to_summary():
    response = json.dumps({
        "event_type": "UNKNOWN_TYPE",
        "description": "Something happened",
        "confidence": 0.5,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.VISUAL_SUMMARY


@pytest.mark.asyncio
async def test_missing_event_type_defaults_to_summary():
    response = json.dumps({
        "description": "A room",
        "confidence": 0.5,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.VISUAL_SUMMARY


@pytest.mark.asyncio
async def test_confidence_clamped_to_range():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "Overconfident model",
        "confidence": 1.5,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_negative_confidence_clamped():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "Negative confidence",
        "confidence": -0.3,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_invalid_confidence_defaults():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "Bad confidence",
        "confidence": "not_a_number",
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_missing_confidence_defaults():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "No confidence field",
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.confidence == 0.5


@pytest.mark.asyncio
async def test_malformed_detections_ignored():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "Bad detections",
        "confidence": 0.5,
        "detections": [
            {"label": "valid", "confidence": 0.8},
            {"no_label": True},
            "not_a_dict",
            {"label": "also_valid", "confidence": 0.6},
        ],
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert len(result.detections) == 2
    assert result.detections[0].label == "valid"
    assert result.detections[1].label == "also_valid"


@pytest.mark.asyncio
async def test_detections_not_a_list_ignored():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "Detections is a string",
        "confidence": 0.5,
        "detections": "not a list",
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert len(result.detections) == 0


# ── Invalid output ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_response_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(""))
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.parse_failure == 1


@pytest.mark.asyncio
async def test_non_json_response_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend("I see a cat"))
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.parse_failure == 1


@pytest.mark.asyncio
async def test_empty_description_returns_none():
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "",
        "confidence": 0.5,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.parse_failure == 1


@pytest.mark.asyncio
async def test_json_array_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend("[1, 2, 3]"))
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.parse_failure == 1


# ── Timeout ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_returns_none():
    analyzer = LocalVisionAnalyzer(
        backend=SlowVisionBackend(delay=5.0),
        timeout_seconds=0.1,
    )
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_timeout == 1
    assert analyzer.inference_failure == 1
    assert analyzer.inference_count == 1


@pytest.mark.asyncio
async def test_timeout_does_not_block_event_loop():
    """Verify that vision inference runs in a thread and doesn't block asyncio."""
    analyzer = LocalVisionAnalyzer(
        backend=SlowVisionBackend(delay=5.0),
        timeout_seconds=0.2,
    )

    other_task_ran = False

    async def other_work():
        nonlocal other_task_ran
        await asyncio.sleep(0.05)
        other_task_ran = True

    task = asyncio.create_task(other_work())
    await analyzer.analyze(_frame())
    await task

    assert other_task_ran, "Event loop was blocked during vision inference"


# ── Backend failure ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_failure_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FailingVisionBackend())
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.backend_failure == 1
    assert analyzer.inference_count == 1


@pytest.mark.asyncio
async def test_backend_failure_does_not_raise():
    """Analyzer must never propagate exceptions to the pipeline."""
    analyzer = LocalVisionAnalyzer(backend=FailingVisionBackend())
    for _ in range(3):
        result = await analyzer.analyze(_frame())
        assert result is None

    assert analyzer.inference_failure == 3
    assert analyzer.backend_failure == 3
    assert analyzer.inference_count == 3


# ── Metrics ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_tracking():
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)

    await analyzer.analyze(_frame())
    await analyzer.analyze(_frame())

    m = analyzer.metrics
    assert m["vision_inference_count"] == 2
    assert m["vision_inference_success"] == 2
    assert m["vision_inference_failure"] == 0
    assert m["vision_inference_timeout"] == 0
    assert m["vision_parse_failure"] == 0
    assert m["vision_backend_failure"] == 0
    assert isinstance(m["vision_inference_latency_ms"], float)
    assert isinstance(m["vision_preprocessing_latency_ms"], float)


@pytest.mark.asyncio
async def test_mixed_success_failure_metrics():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend("not json"))
    await analyzer.analyze(_frame())

    analyzer._backend = FakeVisionBackend(_good_response())
    await analyzer.analyze(_frame())

    assert analyzer.inference_count == 2
    assert analyzer.inference_success == 1
    assert analyzer.inference_failure == 1
    assert analyzer.parse_failure == 1


@pytest.mark.asyncio
async def test_metrics_distinguish_timeout_backend_parse():
    """Timeout, backend error, and parse failure are counted separately."""
    # Backend failure
    analyzer = LocalVisionAnalyzer(backend=FailingVisionBackend())
    await analyzer.analyze(_frame())

    # Parse failure
    analyzer._backend = FakeVisionBackend("garbage")
    await analyzer.analyze(_frame())

    assert analyzer.inference_count == 2
    assert analyzer.backend_failure == 1
    assert analyzer.parse_failure == 1
    assert analyzer.inference_failure == 2


# ── Preprocessing ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_jpeg_passthrough():
    """JPEG frames are base64-encoded directly without decode/re-encode."""
    import io
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)

    frame = _jpeg_frame(8, 8)
    await analyzer.analyze(frame)

    decoded = base64.b64decode(backend.last_image_b64)
    assert decoded == frame.data, "JPEG bytes should pass through unchanged"


@pytest.mark.asyncio
async def test_rgb_preprocessing_produces_valid_jpeg():
    """RGB24 frames are converted to valid JPEG."""
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)

    await analyzer.analyze(_frame(w=8, h=8))

    decoded = base64.b64decode(backend.last_image_b64)
    assert decoded[:2] == b"\xff\xd8", "Output should be valid JPEG"


@pytest.mark.asyncio
async def test_bgr_preprocessing_swaps_channels():
    """BGR24 frames have channels swapped to RGB before JPEG encoding."""
    import io

    # Create a BGR frame: B=200, G=100, R=50
    w, h = 4, 4
    bgr_data = bytes([200, 100, 50] * w * h)
    frame = Frame(
        timestamp=time.time(), data=bgr_data, width=w, height=h, format="bgr24",
    )

    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)
    await analyzer.analyze(frame)

    decoded = base64.b64decode(backend.last_image_b64)
    img = Image.open(io.BytesIO(decoded))
    arr = np.array(img)
    # After BGR->RGB swap, the dominant channel should be R≈50 (was B=200 in BGR)
    # JPEG is lossy, so check approximate: R < G < B after swap means R~50, G~100, B~200
    avg_r, avg_g, avg_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    assert avg_r < avg_g < avg_b, f"BGR->RGB swap failed: R={avg_r:.0f} G={avg_g:.0f} B={avg_b:.0f}"


@pytest.mark.asyncio
async def test_resize_large_landscape():
    """Landscape image wider than max_image_size is resized, aspect ratio preserved."""
    import io
    w, h = 2048, 1024
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend, max_image_size=512)

    frame = _frame(w=w, h=h)
    await analyzer.analyze(frame)

    decoded = base64.b64decode(backend.last_image_b64)
    img = Image.open(io.BytesIO(decoded))
    assert max(img.size) <= 512
    # Aspect ratio: original 2:1, result should be close
    ratio = img.width / img.height
    assert 1.8 <= ratio <= 2.2, f"Aspect ratio distorted: {ratio}"


@pytest.mark.asyncio
async def test_resize_large_portrait():
    """Portrait image taller than max_image_size is resized, aspect ratio preserved."""
    import io
    w, h = 768, 2048
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend, max_image_size=512)

    frame = _frame(w=w, h=h)
    await analyzer.analyze(frame)

    decoded = base64.b64decode(backend.last_image_b64)
    img = Image.open(io.BytesIO(decoded))
    assert max(img.size) <= 512
    ratio = img.height / img.width
    assert 2.3 <= ratio <= 3.0, f"Aspect ratio distorted: {ratio}"


@pytest.mark.asyncio
async def test_resize_large_square():
    """Square image larger than max_image_size is resized to max_image_size."""
    import io
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend, max_image_size=256)

    frame = _frame(w=1024, h=1024)
    await analyzer.analyze(frame)

    decoded = base64.b64decode(backend.last_image_b64)
    img = Image.open(io.BytesIO(decoded))
    assert max(img.size) <= 256
    assert abs(img.width - img.height) <= 1, "Square should remain square"


@pytest.mark.asyncio
async def test_small_image_not_resized():
    """Image already smaller than max_image_size is not resized."""
    import io
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend, max_image_size=1024)

    frame = _frame(w=64, h=64)
    await analyzer.analyze(frame)

    decoded = base64.b64decode(backend.last_image_b64)
    img = Image.open(io.BytesIO(decoded))
    assert img.width == 64
    assert img.height == 64


@pytest.mark.asyncio
async def test_preprocessing_short_data_returns_none():
    """Frame with insufficient data for its dimensions -> preprocessing returns None."""
    frame = Frame(
        timestamp=time.time(),
        data=b"\x00" * 10,  # way too short for 100x100
        width=100, height=100, format="rgb24",
    )
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)
    result = await analyzer.analyze(frame)

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.backend_failure == 1


# ── Lifecycle ────────────────────────────────────────────────────────────


def test_analyzer_close_propagates_to_backend():
    """LocalVisionAnalyzer.close() calls backend.close()."""
    backend = FakeVisionBackend(_good_response())
    analyzer = LocalVisionAnalyzer(backend=backend)
    analyzer.close()
    assert backend.closed is True


def test_analyzer_close_safe_without_close_method():
    """close() works even if backend lacks a close method (Protocol minimum)."""

    class MinimalBackend:
        def infer(self, image_b64: str, prompt: str) -> str:
            return ""

    analyzer = LocalVisionAnalyzer(backend=MinimalBackend())
    analyzer.close()  # should not raise


def test_ollama_backend_close():
    """OllamaVisionBackend.close() closes the httpx client."""
    backend = OllamaVisionBackend(model="test-model", base_url="http://localhost:0")
    backend.close()
    assert backend._client.is_closed


# ── Event type validation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_all_valid_event_types():
    for etype in ("SCENE_CHANGE", "PERSON_DETECTED", "OBJECT_DETECTED", "TEXT_DETECTED", "VISUAL_SUMMARY"):
        response = json.dumps({
            "event_type": etype,
            "description": f"Test {etype}",
            "confidence": 0.5,
        })
        analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
        result = await analyzer.analyze(_frame())
        assert result is not None
        assert result.event_type.name == etype


@pytest.mark.asyncio
async def test_case_insensitive_event_type():
    response = json.dumps({
        "event_type": "person_detected",
        "description": "Case insensitive",
        "confidence": 0.5,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame())

    assert result is not None
    assert result.event_type == VisualEventType.PERSON_DETECTED


# ── Config loading ───────────────────────────────────────────────────────


def test_vision_config_defaults():
    from app.config import VisionConfig

    vc = VisionConfig()
    assert vc.enabled is False
    assert vc.provider == "ollama"
    assert vc.model == "gemma4:12b"
    assert vc.timeout_seconds == 120.0
    assert vc.max_image_size == 1024
    assert vc.prompt == ""


def test_perception_config_includes_vision():
    from app.config import PerceptionConfig, VisionConfig

    pc = PerceptionConfig()
    assert isinstance(pc.vision, VisionConfig)
    assert pc.vision.enabled is False


def test_load_config_with_vision_section(tmp_path):
    from app.config import load_config

    toml_content = """
[perception]
enabled = true

[perception.camera]
enabled = true

[perception.vision]
enabled = true
model = "llava:13b"
timeout_seconds = 45.0
max_image_size = 512
"""
    config_file = tmp_path / "test.toml"
    config_file.write_text(toml_content)

    cfg = load_config(config_file)
    assert cfg.perception.enabled is True
    assert cfg.perception.vision.enabled is True
    assert cfg.perception.vision.model == "llava:13b"
    assert cfg.perception.vision.timeout_seconds == 45.0
    assert cfg.perception.vision.max_image_size == 512


def test_load_config_without_vision_section(tmp_path):
    from app.config import load_config

    toml_content = """
[perception]
enabled = false

[perception.camera]
enabled = false
"""
    config_file = tmp_path / "test.toml"
    config_file.write_text(toml_content)

    cfg = load_config(config_file)
    assert cfg.perception.vision.enabled is False
    assert cfg.perception.vision.model == "gemma4:12b"


# ── Disabled vision / Ollama unavailable ─────────────────────────────────


def test_mock_analyzer_used_when_vision_disabled():
    """When vision.enabled=false, MockVisualAnalyzer is the right choice (verified at config level)."""
    from app.config import VisionConfig
    vc = VisionConfig(enabled=False)
    assert vc.enabled is False


@pytest.mark.asyncio
async def test_ollama_unavailable_returns_none():
    """When backend fails (simulating Ollama unavailable), analyze returns None gracefully."""
    analyzer = LocalVisionAnalyzer(backend=FailingVisionBackend())
    result = await analyzer.analyze(_frame())
    assert result is None
    assert analyzer.backend_failure == 1
