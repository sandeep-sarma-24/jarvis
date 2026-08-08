from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from app.events.types import Event

logger = logging.getLogger(__name__)

EventHandler = Callable[[Event], Awaitable[None]]


class Subscription:
    """Handle returned by EventBus.subscribe — call close() to unsubscribe."""

    def __init__(self, bus: EventBus, key: int) -> None:
        self._bus = bus
        self._key = key

    async def close(self) -> None:
        self._bus._remove(self._key)


class EventBus:
    """In-process asyncio event bus with bounded per-subscriber queues.

    Backpressure: if a subscriber queue is full the *oldest* event in that
    queue is dropped and a warning is logged.  publish() never blocks.
    """

    def __init__(self, queue_size: int = 256) -> None:
        self._queue_size = queue_size
        self._next_key = 0
        # key -> (event_type, handler, queue, task)
        self._subs: dict[int, tuple[type, EventHandler, asyncio.Queue, asyncio.Task]] = {}
        self._running = False
        self.events_published = 0
        self.events_dropped = 0

    async def start(self) -> None:
        self._running = True
        for key, (etype, handler, q, _) in list(self._subs.items()):
            task = asyncio.create_task(self._dispatch(key, handler, q))
            self._subs[key] = (etype, handler, q, task)

    async def shutdown(self) -> None:
        self._running = False
        for _, (_, _, _, task) in self._subs.items():
            if task is not None:
                task.cancel()
        for _, (_, _, _, task) in self._subs.items():
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    def subscribe(self, event_type: type, handler: EventHandler) -> Subscription:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        key = self._next_key
        self._next_key += 1
        task: asyncio.Task | None = None
        if self._running:
            task = asyncio.create_task(self._dispatch(key, handler, q))
        self._subs[key] = (event_type, handler, q, task)
        return Subscription(self, key)

    async def publish(self, event: Event) -> None:
        self.events_published += 1
        for key, (etype, handler, q, task) in self._subs.items():
            if not isinstance(event, etype):
                continue
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.events_dropped += 1
                logger.warning(
                    "subscriber %d queue full — dropped oldest event (event_id=%s)",
                    key,
                    event.id,
                )
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                self.events_dropped += 1

    def _remove(self, key: int) -> None:
        entry = self._subs.pop(key, None)
        if entry is not None:
            task = entry[3]
            if task is not None:
                task.cancel()

    async def _dispatch(self, key: int, handler: EventHandler, q: asyncio.Queue) -> None:
        try:
            while True:
                event = await q.get()
                try:
                    await handler(event)
                except Exception:
                    logger.exception("subscriber %d handler failed on event %s", key, event.id)
        except asyncio.CancelledError:
            return
