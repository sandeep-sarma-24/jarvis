import asyncio

import pytest

from app.events.bus import EventBus
from app.events.types import Event, Modality, SpeechEvent


@pytest.fixture
async def bus():
    b = EventBus(queue_size=4)
    await b.start()
    yield b
    await b.shutdown()


async def test_publish_subscribe(bus: EventBus):
    received: list[Event] = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe(SpeechEvent, handler)
    await asyncio.sleep(0.01)

    event = SpeechEvent(source=Modality.AUDIO, transcript="test")
    await bus.publish(event)
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].transcript == "test"


async def test_type_filtering(bus: EventBus):
    speech_received: list[Event] = []
    all_received: list[Event] = []

    async def speech_handler(event: Event):
        speech_received.append(event)

    async def all_handler(event: Event):
        all_received.append(event)

    bus.subscribe(SpeechEvent, speech_handler)
    bus.subscribe(Event, all_handler)
    await asyncio.sleep(0.01)

    base = Event(source=Modality.SYSTEM)
    speech = SpeechEvent(source=Modality.AUDIO, transcript="hi")

    await bus.publish(base)
    await bus.publish(speech)
    await asyncio.sleep(0.05)

    assert len(speech_received) == 1
    assert len(all_received) == 2


async def test_multiple_subscribers(bus: EventBus):
    a: list[Event] = []
    b: list[Event] = []

    async def ha(e: Event):
        a.append(e)

    async def hb(e: Event):
        b.append(e)

    bus.subscribe(Event, ha)
    bus.subscribe(Event, hb)
    await asyncio.sleep(0.01)

    await bus.publish(Event())
    await asyncio.sleep(0.05)

    assert len(a) == 1
    assert len(b) == 1


async def test_subscriber_failure_isolation(bus: EventBus):
    good: list[Event] = []

    async def bad_handler(event: Event):
        raise ValueError("boom")

    async def good_handler(event: Event):
        good.append(event)

    bus.subscribe(Event, bad_handler)
    bus.subscribe(Event, good_handler)
    await asyncio.sleep(0.01)

    await bus.publish(Event())
    await asyncio.sleep(0.05)

    assert len(good) == 1


async def test_queue_saturation():
    bus = EventBus(queue_size=2)
    received: list[Event] = []

    async def slow_handler(event: Event):
        await asyncio.sleep(0.1)
        received.append(event)

    bus.subscribe(Event, slow_handler)
    await bus.start()

    for _ in range(5):
        await bus.publish(Event())

    assert bus.events_dropped > 0
    await bus.shutdown()


async def test_unsubscribe(bus: EventBus):
    received: list[Event] = []

    async def handler(event: Event):
        received.append(event)

    sub = bus.subscribe(Event, handler)
    await asyncio.sleep(0.01)

    await bus.publish(Event())
    await asyncio.sleep(0.05)
    assert len(received) == 1

    await sub.close()
    await bus.publish(Event())
    await asyncio.sleep(0.05)
    assert len(received) == 1


async def test_shutdown_is_clean(bus: EventBus):
    async def handler(event: Event):
        pass

    bus.subscribe(Event, handler)
    await bus.shutdown()
