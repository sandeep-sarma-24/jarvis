"""Integration test: SpeechEvent → EventBus → Attention subscriber → AttentionDecision."""
import asyncio

import pytest

from app.attention.engine import AttentionDecision, AttentionEngine
from app.events.bus import EventBus
from app.events.types import Event, Modality, SpeechEvent


def _speech(transcript: str) -> SpeechEvent:
    return SpeechEvent(
        session_id="integration-test",
        correlation_id="test-corr",
        source=Modality.AUDIO,
        transcript=transcript,
        duration_seconds=1.5,
    )


async def test_attention_subscriber_receives_events():
    """Full EventBus integration: publish SpeechEvent → attention handler evaluates."""
    bus = EventBus(queue_size=16)
    engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
    decisions: list[AttentionDecision] = []

    async def attention_handler(event: Event) -> None:
        if isinstance(event, SpeechEvent):
            d = engine.evaluate(event)
            decisions.append(d)

    bus.subscribe(SpeechEvent, attention_handler)
    await bus.start()

    await bus.publish(_speech("Hey Jarvis, what is the weather?"))
    await asyncio.sleep(0.05)

    assert len(decisions) == 1
    assert decisions[0].should_reason is True
    assert decisions[0].salience >= 0.9

    await bus.publish(_speech("I'll call you tomorrow."))
    await asyncio.sleep(0.05)

    assert len(decisions) == 2
    assert decisions[1].should_reason is False

    await bus.shutdown()


async def test_attention_does_not_block_other_subscribers():
    """Attention subscriber must not prevent console subscriber from receiving events."""
    bus = EventBus(queue_size=16)
    engine = AttentionEngine(assistant_names=["jarvis"])
    console_received: list[SpeechEvent] = []
    attention_decisions: list[AttentionDecision] = []

    async def console_handler(event: Event) -> None:
        if isinstance(event, SpeechEvent):
            console_received.append(event)

    async def attention_handler(event: Event) -> None:
        if isinstance(event, SpeechEvent):
            d = engine.evaluate(event)
            attention_decisions.append(d)

    bus.subscribe(SpeechEvent, console_handler)
    bus.subscribe(SpeechEvent, attention_handler)
    await bus.start()

    await bus.publish(_speech("Hey Jarvis, help me"))
    await asyncio.sleep(0.05)

    assert len(console_received) == 1
    assert len(attention_decisions) == 1

    await bus.shutdown()


async def test_attention_counters_via_eventbus():
    """Counters update correctly when events flow through the bus."""
    bus = EventBus(queue_size=16)
    engine = AttentionEngine(assistant_names=["jarvis"])

    async def handler(event: Event) -> None:
        if isinstance(event, SpeechEvent):
            engine.evaluate(event)

    bus.subscribe(SpeechEvent, handler)
    await bus.start()

    await bus.publish(_speech("Hey Jarvis"))
    await bus.publish(_speech("background noise"))
    await bus.publish(_speech(""))
    await asyncio.sleep(0.05)

    c = engine.counters
    assert c["evaluations"] == 3
    assert c["reason_true"] >= 1
    assert c["reason_false"] >= 1

    await bus.shutdown()
