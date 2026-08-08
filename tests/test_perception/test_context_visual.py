"""Tests for visual context integration with ContextBuilder."""
from __future__ import annotations

import time
from io import StringIO
from pathlib import Path

import pytest

from app.context.builder import ContextBuilder
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(
        source=Modality.AUDIO,
        transcript=text,
        duration_seconds=1.0,
    )


def _builder(**kwargs) -> ContextBuilder:
    defaults = {
        "max_messages": 8,
        "max_characters": 6000,
        "max_visual_events": 5,
        "max_visual_context_characters": 500,
    }
    defaults.update(kwargs)
    return ContextBuilder(**defaults)


def test_visual_context_included_in_build():
    builder = _builder()
    wm = WorkingMemory(max_size=5)

    ctx = builder.build(
        _speech("What do you see?"),
        wm,
        visual_context=["A person entered the room 3 seconds ago"],
    )

    assert len(ctx.visual_context) == 1
    assert "person entered" in ctx.visual_context[0]


def test_visual_context_in_messages():
    builder = _builder()
    wm = WorkingMemory(max_size=5)

    ctx = builder.build(
        _speech("Describe the room"),
        wm,
        visual_context=["Desk with monitor and keyboard"],
    )

    messages = builder.to_messages(ctx)
    visual_msgs = [m for m in messages if "visual context" in m.get("content", "").lower()]
    assert len(visual_msgs) == 1
    assert "Desk with monitor" in visual_msgs[0]["content"]


def test_visual_context_bounded():
    builder = _builder(max_visual_events=2, max_visual_context_characters=50)
    wm = WorkingMemory(max_size=5)

    ctx = builder.build(
        _speech("What happened?"),
        wm,
        visual_context=[
            "First visual event description",
            "Second visual event description",
            "Third visual event should be excluded",
        ],
    )

    assert len(ctx.visual_context) <= 2


def test_visual_context_respects_character_budget():
    builder = _builder(max_visual_context_characters=30)
    wm = WorkingMemory(max_size=5)

    ctx = builder.build(
        _speech("test"),
        wm,
        visual_context=[
            "Short",
            "This is a much longer description that should exceed the budget",
        ],
    )

    total_vis_chars = sum(len(v) for v in ctx.visual_context)
    assert total_vis_chars <= 30


def test_no_visual_context_backward_compatible():
    builder = _builder()
    wm = WorkingMemory(max_size=5)

    ctx = builder.build(_speech("Hello"), wm)

    assert ctx.visual_context == ()
    messages = builder.to_messages(ctx)
    visual_msgs = [m for m in messages if "visual context" in m.get("content", "").lower()]
    assert len(visual_msgs) == 0


def test_visual_context_does_not_starve_speech():
    """Visual context has its own sub-budget — doesn't consume all characters."""
    builder = _builder(max_visual_context_characters=100)
    wm = WorkingMemory(max_size=5)

    ctx = builder.build(
        _speech("Tell me about the project"),
        wm,
        conversation_messages=[
            {"role": "user", "content": "Previous question about code"},
            {"role": "assistant", "content": "I explained the architecture"},
        ],
        memory_context=["User prefers Python"],
        visual_context=["Person sitting at desk"],
    )

    assert len(ctx.conversation_messages) > 0
    assert len(ctx.memory_context) > 0
    assert len(ctx.visual_context) > 0


def test_total_characters_includes_visual():
    builder = _builder()
    wm = WorkingMemory(max_size=5)

    ctx_no_vis = builder.build(_speech("test"), wm)
    ctx_with_vis = builder.build(
        _speech("test"), wm, visual_context=["A person is present"],
    )

    assert ctx_with_vis.total_characters > ctx_no_vis.total_characters
