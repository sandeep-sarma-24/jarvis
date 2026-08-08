"""Tests for VisualAnalyzer, ChangeDetector, and MockVisualAnalyzer."""
from __future__ import annotations

import time

import pytest

from app.perception.types import Frame, VisualEventType
from app.perception.visual import ChangeDetector, MockVisualAnalyzer


def _frame(pixel: int = 0, width: int = 4, height: int = 4) -> Frame:
    return Frame(
        timestamp=time.time(),
        data=bytes([pixel] * (width * height * 3)),
        width=width,
        height=height,
        format="rgb24",
    )


@pytest.mark.asyncio
async def test_mock_analyzer_returns_analysis():
    analyzer = MockVisualAnalyzer()
    frame = _frame(pixel=5)  # 5 % 5 == 0 → person detected
    result = await analyzer.analyze(frame)

    assert result is not None
    assert result.event_type == VisualEventType.PERSON_DETECTED
    assert result.confidence > 0
    assert len(result.detections) == 1
    assert result.detections[0].label == "person"


@pytest.mark.asyncio
async def test_mock_analyzer_scene_change():
    analyzer = MockVisualAnalyzer()
    frame = _frame(pixel=3)  # 3 % 3 == 0 and 3 % 5 != 0 → scene change
    result = await analyzer.analyze(frame)

    assert result is not None
    assert result.event_type == VisualEventType.SCENE_CHANGE


@pytest.mark.asyncio
async def test_mock_analyzer_summary():
    analyzer = MockVisualAnalyzer()
    frame = _frame(pixel=1)  # 1 % 5 != 0 and 1 % 3 != 0 → summary
    result = await analyzer.analyze(frame)

    assert result is not None
    assert result.event_type == VisualEventType.VISUAL_SUMMARY


@pytest.mark.asyncio
async def test_mock_analyzer_empty_frame():
    analyzer = MockVisualAnalyzer()
    frame = Frame(timestamp=time.time(), data=b"", width=0, height=0, format="rgb24")
    result = await analyzer.analyze(frame)
    assert result is None


@pytest.mark.asyncio
async def test_mock_analyzer_counter():
    analyzer = MockVisualAnalyzer()
    for _ in range(3):
        await analyzer.analyze(_frame(pixel=1))
    assert analyzer.analyses_performed == 3


def test_change_detector_first_frame_always_changed():
    cd = ChangeDetector(change_threshold=0.05)
    assert cd.has_changed(_frame(pixel=100)) is True
    assert cd.changes_detected == 1


def test_change_detector_identical_frames_no_change():
    cd = ChangeDetector(change_threshold=0.05)
    cd.has_changed(_frame(pixel=100))
    assert cd.has_changed(_frame(pixel=100)) is False
    assert cd.changes_detected == 1


def test_change_detector_different_frames_changed():
    cd = ChangeDetector(change_threshold=0.05)
    cd.has_changed(_frame(pixel=0))
    assert cd.has_changed(_frame(pixel=200)) is True
    assert cd.changes_detected == 2


def test_change_detector_metrics():
    cd = ChangeDetector(change_threshold=0.05)
    cd.has_changed(_frame(pixel=0))
    cd.has_changed(_frame(pixel=0))
    cd.has_changed(_frame(pixel=255))

    assert cd.checks_performed == 3
    assert cd.changes_detected == 2
