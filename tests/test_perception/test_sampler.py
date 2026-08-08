"""Tests for FrameSampler: latest-frame policy, interval, freshness, bounded queue."""
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
    sampler.next_frame()

    sampler.submit(_frame(pixel=2))
    result = sampler.next_frame()
    assert result is None


def test_interval_does_not_drain_queue():
    """When interval hasn't elapsed, the queue is untouched — frames stay for later."""
    sampler = FrameSampler(analysis_interval_ms=10000.0, max_frame_age_seconds=5.0, max_queue_size=4)
    sampler.submit(_frame(pixel=1))
    sampler.next_frame()  # first passes

    sampler.submit(_frame(pixel=2))
    sampler.submit(_frame(pixel=3))
    sampler.next_frame()  # interval not elapsed → None, queue untouched
    assert sampler.queue_depth == 2
    sampler.next_frame()  # poll again → still None, no double-counting
    assert sampler.queue_depth == 2


def test_stale_frame_dropped():
    sampler = FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=0.001)
    f = _frame(age_offset=1.0)
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


def test_latest_frame_selected_from_queue():
    """next_frame() returns the newest fresh frame, not the oldest."""
    sampler = FrameSampler(
        analysis_interval_ms=0, max_frame_age_seconds=5.0, max_queue_size=4,
    )
    sampler.submit(_frame(pixel=10))
    sampler.submit(_frame(pixel=20))
    sampler.submit(_frame(pixel=30))

    result = sampler.next_frame()
    assert result is not None
    assert result.data[0] == 30  # newest frame selected
    assert sampler.frames_dropped_superseded == 2  # older fresh frames discarded
    assert sampler.frames_sampled == 1


def test_latest_frame_with_queue_size_1():
    """When queue is size 1, only the newest frame survives (via eviction)."""
    sampler = FrameSampler(
        analysis_interval_ms=0, max_frame_age_seconds=5.0, max_queue_size=1,
    )
    sampler.submit(_frame(pixel=10))
    sampler.submit(_frame(pixel=20))
    sampler.submit(_frame(pixel=30))

    result = sampler.next_frame()
    assert result is not None
    assert result.data[0] == 30


def test_stale_frames_skipped_fresh_frame_returned():
    """Stale frames in queue are skipped; newest fresh frame is returned."""
    sampler = FrameSampler(
        analysis_interval_ms=0, max_frame_age_seconds=0.5, max_queue_size=4,
    )
    sampler.submit(_frame(pixel=1, age_offset=2.0))  # stale
    sampler.submit(_frame(pixel=2, age_offset=2.0))  # stale
    sampler.submit(_frame(pixel=3))                   # fresh

    result = sampler.next_frame()
    assert result is not None
    assert result.data[0] == 3
    assert sampler.frames_dropped_stale == 2
    assert sampler.frames_sampled == 1


def test_empty_queue_returns_none():
    sampler = FrameSampler(analysis_interval_ms=0)
    result = sampler.next_frame()
    assert result is None


def test_metrics_no_double_counting():
    """Each frame ends up in exactly one terminal category."""
    sampler = FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=0.5, max_queue_size=2)
    sampler.submit(_frame(pixel=1, age_offset=2.0))  # will be stale
    sampler.submit(_frame(pixel=2))                   # fresh, but superseded
    sampler.submit(_frame(pixel=3))                   # fresh, newest → sampled (evicts pixel=1)

    result = sampler.next_frame()
    assert result is not None
    assert result.data[0] == 3

    m = sampler.metrics
    assert m["frames_received"] == 3
    assert m["frames_sampled"] == 1
    assert m["frames_evicted"] == 1           # pixel=1 evicted by pixel=3
    assert m["frames_dropped_superseded"] == 1  # pixel=2 superseded by pixel=3
    assert m["frames_dropped_stale"] == 0     # pixel=1 was evicted, not stale-dropped


def test_all_stale_returns_none():
    """If every frame in the queue is stale, next_frame returns None."""
    sampler = FrameSampler(analysis_interval_ms=0, max_frame_age_seconds=0.01)
    sampler.submit(_frame(pixel=1, age_offset=1.0))
    sampler.submit(_frame(pixel=2, age_offset=1.0))

    result = sampler.next_frame()
    assert result is None
    assert sampler.frames_dropped_stale == 2
    assert sampler.frames_sampled == 0
