"""Tests for EventDeduplicator — visual event suppression within a time window."""
from __future__ import annotations

import time

import pytest

from app.perception.visual import EventDeduplicator


def test_first_event_not_duplicate():
    dd = EventDeduplicator(dedup_window_seconds=30.0)
    assert dd.is_duplicate("SCENE_CHANGE", "Person entered room") is False
    assert dd.events_suppressed == 0


def test_same_event_within_window_is_duplicate():
    dd = EventDeduplicator(dedup_window_seconds=30.0)
    dd.is_duplicate("SCENE_CHANGE", "Person entered room")
    assert dd.is_duplicate("SCENE_CHANGE", "Person entered room") is True
    assert dd.events_suppressed == 1


def test_different_type_not_duplicate():
    dd = EventDeduplicator(dedup_window_seconds=30.0)
    dd.is_duplicate("SCENE_CHANGE", "Person entered room")
    assert dd.is_duplicate("PERSON_DETECTED", "Person entered room") is False


def test_different_description_not_duplicate():
    dd = EventDeduplicator(dedup_window_seconds=30.0)
    dd.is_duplicate("SCENE_CHANGE", "Person entered room")
    assert dd.is_duplicate("SCENE_CHANGE", "Person left room") is False


def test_expired_events_pruned():
    dd = EventDeduplicator(dedup_window_seconds=0.001)  # 1ms window
    dd.is_duplicate("SCENE_CHANGE", "Person entered room")
    time.sleep(0.01)  # wait for expiry
    assert dd.is_duplicate("SCENE_CHANGE", "Person entered room") is False


def test_metrics():
    dd = EventDeduplicator(dedup_window_seconds=30.0)
    dd.is_duplicate("A", "x")
    dd.is_duplicate("A", "x")
    dd.is_duplicate("B", "y")

    assert dd.events_checked == 3
    assert dd.events_suppressed == 1
