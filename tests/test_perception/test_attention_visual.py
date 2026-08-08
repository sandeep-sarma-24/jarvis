"""Tests for visual event routing through AttentionEngine."""
from __future__ import annotations

import pytest

from app.attention.engine import AttentionDecision, AttentionEngine
from app.events.types import Modality, VisualEvent
from app.memory.working import WorkingMemory


def _engine() -> AttentionEngine:
    return AttentionEngine(
        assistant_names=["jarvis"],
        reason_threshold=0.4,
        working_memory=WorkingMemory(max_size=10),
    )


def test_high_salience_person_detected():
    engine = _engine()
    event = VisualEvent(
        source=Modality.VISUAL,
        event_type="PERSON_DETECTED",
        description="Someone entered the room",
        confidence=0.9,
    )
    decision = engine.evaluate_visual(event)
    assert decision.should_reason is True
    assert decision.tier == "visual"
    assert decision.salience >= 0.7


def test_high_salience_scene_change():
    engine = _engine()
    event = VisualEvent(
        source=Modality.VISUAL,
        event_type="SCENE_CHANGE",
        description="Scene changed significantly",
        confidence=0.85,
    )
    decision = engine.evaluate_visual(event)
    assert decision.should_reason is True


def test_low_salience_visual_summary():
    engine = _engine()
    event = VisualEvent(
        source=Modality.VISUAL,
        event_type="VISUAL_SUMMARY",
        description="Desk with coffee cup",
        confidence=0.7,
    )
    decision = engine.evaluate_visual(event)
    assert decision.should_reason is False


def test_low_confidence_person_not_worthy():
    engine = _engine()
    event = VisualEvent(
        source=Modality.VISUAL,
        event_type="PERSON_DETECTED",
        description="Possible person in shadow",
        confidence=0.3,
    )
    decision = engine.evaluate_visual(event)
    assert decision.should_reason is False


def test_visual_counters():
    engine = _engine()

    high = VisualEvent(source=Modality.VISUAL, event_type="PERSON_DETECTED",
                       description="Person", confidence=0.9)
    low = VisualEvent(source=Modality.VISUAL, event_type="VISUAL_SUMMARY",
                      description="Desk", confidence=0.5)

    engine.evaluate_visual(high)
    engine.evaluate_visual(low)
    engine.evaluate_visual(low)

    counters = engine.counters
    assert counters["visual_evaluations"] == 3
    assert counters["visual_reason_true"] == 1
    assert counters["visual_reason_false"] == 2


def test_visual_does_not_affect_speech_counters():
    engine = _engine()
    event = VisualEvent(source=Modality.VISUAL, event_type="SCENE_CHANGE",
                        description="Change", confidence=0.9)
    engine.evaluate_visual(event)

    assert engine.counters["evaluations"] == 0  # speech counter untouched
    assert engine.counters["visual_evaluations"] == 1
