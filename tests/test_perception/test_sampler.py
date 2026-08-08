"""Tests for FrameSampler: interval, freshness, bounded queue, latest-frame policy."""
from __future__ import annotations

import time

import pytest

from app.perception.sampler import FrameSampler
from app.perception.types import Frame


def _frame(*, age_offset: float = 0.0, pixel: int = 0) -> Frame:
    return Frame(
        timestamp=time.time() - age_offset,
        data=bytes([pixel] * 12),
        width=2,
        height=2,
        format="rgb24",
    )


def test_first_frame_passes_immediately():
    sampler = FrameSampler(analysis_interval_ms=1000.0, max_frame_age_seconds=5.0)
    f = _frame()
    sampler.submit(f)
    result = sampler.next_frame()
    assert result is not None
    assert sampler.frames_sampled == 1


def test_interval_blocks_second_frame():
    sampler = FrameSampler(analysis_interval_ms=10000.0, max_frame_age_seconds=5.0)
    sampler.submit(_frame(pixel=1))
    sampler.next_frame()  # first passes

    sampler.submit(_frame(pixel=2))
    result = sampler.next_frame()
    assert result is None  # too soon


def test_interval_increments_dropped_interval():
    """When interval hasn't elapsed, frames_dropped_interval is incremented."""
    sampler = FrameSampler(analysis_interval_ms=10000.0, max_frame_age_seconds=5.0, max_queue_size=4)
    sampler.submit(_frame(pixel=1))
    sampler.next_frame()  # first passes

    sampler.submit(_frame(pixel=2))
    sampler.submit(_frame(pixel=3))
    result = sampler.next_frame()
    assert result is None
    assert sampler.frames_dropped_interval == 2  # 2 frames in queue when interval gate fires


def test_stale_frame_dropped():
    sampler = FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=0.001)
    f = _frame(age_offset=1.0)  # 1 second old
    sampler.submit(f)
    result = sampler.next_frame()
    assert result is None
    assert sampler.frames_dropped_stale == 1


def test_bounded_queue_evicts_oldest():
    sampler = FrameSampler(
        analysis_interval_ms=0, max_frame_age_seconds=5.0, max_queue_size=2,
    )
    f1 = _frame(pixel=1)
    f2 = _frame(pixel=2)
    f3 = _frame(pixel=3)

    sampler.submit(f1)
    sampler.submit(f2)
    assert sampler.queue_depth == 2
    assert sampler.frames_evicted == 0

    sampler.submit(f3)  # evicts f1
    assert sampler.frames_evicted == 1

    result = sampler.next_frame()
    assert result is not None
    assert result.data[0] == 2  # f2 is now oldest in queue


def test_latest_frame_policy():
    """When queue is size 1, only the newest frame survives."""
    sampler = FrameSampler(
        analysis_interval_ms=0, max_frame_age_seconds=5.0, max_queue_size=1,
    )
    sampler.submit(_frame(pixel=10))
    sampler.submit(_frame(pixel=20))
    sampler.submit(_frame(pixel=30))

    result = sampler.next_frame()
    assert result is not None
    assert result.data[0] == 30  # only latest survived


def test_empty_queue_returns_none():
    sampler = FrameSampler(analysis_interval_ms=0)
    result = sampler.next_frame()
    assert result is None


def test_metrics_tracking():
    sampler = FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=5.0, max_queue_size=2)
    sampler.submit(_frame(pixel=1))
    sampler.submit(_frame(pixel=2))
    sampler.submit(_frame(pixel=3))
    sampler.next_frame()

    m = sampler.metrics
    assert m["frames_received"] == 3
    assert m["frames_sampled"] == 1
    assert m["frames_evicted"] == 1
