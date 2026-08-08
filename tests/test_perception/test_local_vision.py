"""Tests for LocalVisionAnalyzer: structured output parsing, timeouts, failures.

All tests use a FakeVisionBackend — no real model or Ollama instance required.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.perception.local_vision import LocalVisionAnalyzer, VisionBackend
from app.perception.types import Frame, VisualEventType


def _frame(*, pixel: int = 128, w: int = 4, h: int = 4) -> Frame:
    return Frame(
        timestamp=time.time(),
        data=bytes([pixel] * w * h * 3),
        width=w,
        height=h,
        format="rgb24",
    )


class FakeVisionBackend:
    """Returns a canned response string. Implements VisionBackend protocol."""

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.call_count = 0

    def infer(self, image_b64: str, prompt: str) -> str:
        self.call_count += 1
        return self.response


class SlowVisionBackend:
    """Backend that sleeps, for timeout testing."""

    def __init__(self, delay: float = 10.0) -> None:
        self._delay = delay

    def infer(self, image_b64: str, prompt: str) -> str:
        time.sleep(self._delay)
        return '{"event_type": "VISUAL_SUMMARY", "description": "too slow", "confidence": 0.5}'


class FailingVisionBackend:
    """Backend that raises on every call."""

    def infer(self, image_b64: str, prompt: str) -> str:
        raise ConnectionError("model unavailable")


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


@pytest.mark.asyncio
async def test_non_json_response_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend("I see a cat"))
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1


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


@pytest.mark.asyncio
async def test_json_array_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend("[1, 2, 3]"))
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1


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


# ── Backend failure ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backend_failure_returns_none():
    analyzer = LocalVisionAnalyzer(backend=FailingVisionBackend())
    result = await analyzer.analyze(_frame())

    assert result is None
    assert analyzer.inference_failure == 1
    assert analyzer.inference_count == 1


@pytest.mark.asyncio
async def test_backend_failure_does_not_raise():
    """Analyzer must never propagate exceptions to the pipeline."""
    analyzer = LocalVisionAnalyzer(backend=FailingVisionBackend())
    for _ in range(3):
        result = await analyzer.analyze(_frame())
        assert result is None

    assert analyzer.inference_failure == 3
    assert analyzer.inference_count == 3


# ── Metrics ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_tracking():
    good_response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "A desk",
        "confidence": 0.7,
    })
    backend = FakeVisionBackend(good_response)
    analyzer = LocalVisionAnalyzer(backend=backend)

    await analyzer.analyze(_frame())
    await analyzer.analyze(_frame())

    m = analyzer.metrics
    assert m["vision_inference_count"] == 2
    assert m["vision_inference_success"] == 2
    assert m["vision_inference_failure"] == 0
    assert m["vision_inference_timeout"] == 0
    assert isinstance(m["vision_inference_latency_ms"], float)
    assert isinstance(m["vision_preprocessing_latency_ms"], float)


@pytest.mark.asyncio
async def test_mixed_success_failure_metrics():
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend("not json"))
    await analyzer.analyze(_frame())

    valid = json.dumps({"event_type": "VISUAL_SUMMARY", "description": "ok", "confidence": 0.5})
    analyzer._backend = FakeVisionBackend(valid)
    await analyzer.analyze(_frame())

    assert analyzer.inference_count == 2
    assert analyzer.inference_success == 1
    assert analyzer.inference_failure == 1


# ── Preprocessing ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preprocessing_handles_small_frames():
    """Small frames (like test 4x4) should not crash preprocessing."""
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "Tiny frame",
        "confidence": 0.5,
    })
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(_frame(w=2, h=2))

    assert result is not None


@pytest.mark.asyncio
async def test_preprocessing_handles_jpeg_format():
    """JPEG-format frames skip numpy reshape and go straight to base64."""
    response = json.dumps({
        "event_type": "VISUAL_SUMMARY",
        "description": "JPEG frame",
        "confidence": 0.5,
    })
    f = Frame(
        timestamp=time.time(),
        data=b"\xff\xd8\xff\xe0" + b"\x00" * 100,
        width=4,
        height=4,
        format="jpeg",
    )
    analyzer = LocalVisionAnalyzer(backend=FakeVisionBackend(response))
    result = await analyzer.analyze(f)

    assert result is not None


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


# ── Custom prompt ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_prompt_passed_to_backend():
    """When a custom prompt is set, it's forwarded to the backend."""

    class PromptCapturingBackend:
        def __init__(self) -> None:
            self.last_prompt = ""

        def infer(self, image_b64: str, prompt: str) -> str:
            self.last_prompt = prompt
            return json.dumps({"event_type": "VISUAL_SUMMARY", "description": "ok", "confidence": 0.5})

    backend = PromptCapturingBackend()
    analyzer = LocalVisionAnalyzer(backend=backend, prompt="Describe this scene.")
    await analyzer.analyze(_frame())

    assert backend.last_prompt == "Describe this scene."


# ── Config loading ───────────────────────────────────────────────────────


def test_vision_config_defaults():
    from app.config import VisionConfig

    vc = VisionConfig()
    assert vc.enabled is False
    assert vc.provider == "ollama"
    assert vc.model == "gemma4:12b"
    assert vc.timeout_seconds == 30.0
    assert vc.max_image_size == 1024


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
