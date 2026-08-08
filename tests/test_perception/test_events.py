"""Tests for VisualEvent type and EventBus integration."""
from __future__ import annotations

import pytest

from app.events.bus import EventBus
from app.events.types import Event, Modality, SpeechEvent, VisualEvent


def test_visual_event_fields():
    e = VisualEvent(
        source=Modality.VISUAL,
        event_type="PERSON_DETECTED",
        description="Person near desk",
        confidence=0.9,
        metadata={"test": True},
    )
    assert e.source == Modality.VISUAL
    assert e.event_type == "PERSON_DETECTED"
    assert e.description == "Person near desk"
    assert e.confidence == 0.9
    assert e.id  # auto-generated


def test_visual_event_is_event():
    e = VisualEvent()
    assert isinstance(e, Event)


@pytest.mark.asyncio
async def test_visual_event_bus_routing():
    """VisualEvents reach visual subscribers but not speech subscribers."""
    bus = EventBus(queue_size=16)

    visual_received = []
    speech_received = []

    async def visual_handler(event: Event) -> None:
        visual_received.append(event)

    async def speech_handler(event: Event) -> None:
        speech_received.append(event)

    bus.subscribe(VisualEvent, visual_handler)
    bus.subscribe(SpeechEvent, speech_handler)
    await bus.start()

    ve = VisualEvent(
        source=Modality.VISUAL,
        event_type="SCENE_CHANGE",
        description="Scene changed",
    )
    await bus.publish(ve)

    # Give dispatch tasks time to run
    import asyncio
    await asyncio.sleep(0.05)

    assert len(visual_received) == 1
    assert len(speech_received) == 0

    await bus.shutdown()


@pytest.mark.asyncio
async def test_speech_event_does_not_reach_visual_subscriber():
    bus = EventBus(queue_size=16)

    visual_received = []

    async def visual_handler(event: Event) -> None:
        visual_received.append(event)

    bus.subscribe(VisualEvent, visual_handler)
    await bus.start()

    se = SpeechEvent(
        source=Modality.AUDIO,
        transcript="hello",
    )
    await bus.publish(se)

    import asyncio
    await asyncio.sleep(0.05)

    assert len(visual_received) == 0

    await bus.shutdown()
