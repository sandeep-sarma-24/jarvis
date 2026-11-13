"""Scene tracking and VLM gating on top of detector output.

SceneTracker and VlmGate are pure logic with injected timestamps (no model, no camera).
SceneWatcher drives them: camera -> detector thread -> tracker -> gate -> VLM, off the event loop.
Thresholds follow the vlm_report gating policy.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from app.perception.types import Entity, Frame

logger = logging.getLogger(__name__)


@dataclass
class _Track:
    label: str
    first: float
    last: float
    hits: int = 0          # consecutive frames at conf >= min_conf
    confirmed: bool = False


@dataclass(frozen=True)
class SceneEvent:
    kind: str              # "entered" | "left"
    track_id: int
    label: str
    t: float
    new_class: bool = False  # label had no confirmed track for class_memory seconds

    def __str__(self) -> str:
        return f"{self.label} {self.kind}"


class SceneTracker:
    """Debounced enter/leave and per-label counts from per-frame tracked entities."""

    def __init__(self, *, min_conf=0.5, min_frames=3, person_frames=2, min_age=1.0,
                 left_after=1.0, class_memory=60.0, recent_window=30.0) -> None:
        self._min_conf, self._min_frames, self._person_frames = min_conf, min_frames, person_frames
        self._min_age, self._left_after, self._class_memory = min_age, left_after, class_memory
        self._recent_window = recent_window
        self._tracks: dict[int, _Track] = {}
        self._label_seen: dict[str, float] = {}
        self._recent: deque[tuple[float, str]] = deque(maxlen=20)

    def update(self, now: float, entities: list[Entity]) -> list[SceneEvent]:
        events: list[SceneEvent] = []
        seen = set()
        for e in entities:
            seen.add(e.track_id)
            tr = self._tracks.setdefault(e.track_id, _Track(e.label, now, now))
            tr.label, tr.last = e.label, now
            tr.hits = tr.hits + 1 if e.conf >= self._min_conf else 0
            person = e.label == "person"
            if not tr.confirmed and tr.hits >= (self._person_frames if person else self._min_frames) \
                    and (person or now - tr.first >= self._min_age):
                tr.confirmed = True
                seen_at = self._label_seen.get(e.label)
                new = seen_at is None or now - seen_at > self._class_memory
                self._label_seen[e.label] = now
                events.append(SceneEvent("entered", e.track_id, e.label, now, new))
        for tid, tr in list(self._tracks.items()):
            if tr.confirmed and tid in seen:
                self._label_seen[tr.label] = now
            elif tid not in seen and now - tr.last > self._left_after:
                del self._tracks[tid]
                if tr.confirmed:
                    events.append(SceneEvent("left", tid, tr.label, now))
        self._recent.extend((now, str(ev)) for ev in events)
        return events

    def ids(self) -> frozenset[int]:
        return frozenset(i for i, t in self._tracks.items() if t.confirmed)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self._tracks.values():
            if t.confirmed:
                out[t.label] = out.get(t.label, 0) + 1
        return out

    def summary(self, now: float) -> str:
        counts = self.counts()
        body = ", ".join(f"{l}×{n}" for l, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) \
            or "nothing detected"
        recent = [f"{text} {max(0, round(now - t))}s ago" for t, text in self._recent
                  if now - t <= self._recent_window][-2:]
        return f"{body} ({'; '.join(recent)})" if recent else body


@dataclass(frozen=True)
class VlmRequest:
    kind: str                        # new_class | scene_change | count_change | refresh
    track_ids: tuple[int, ...] = ()  # entities that changed (exactly one -> crop request)


class VlmGate:
    """Decides when the VLM may run. Drops (never queues) when busy, cooling down or rate limited."""

    def __init__(self, *, cooldown=3.0, max_per_min=10, jaccard=0.4, count_stable=2.0,
                 motion_hold=1.0, refresh_person=60.0, refresh_empty=180.0, start=0.0) -> None:
        self._cooldown, self._max_per_min, self._jaccard = cooldown, max_per_min, jaccard
        self._count_stable, self._motion_hold = count_stable, motion_hold
        self._refresh_person, self._refresh_empty = refresh_person, refresh_empty
        self._last_call = start
        self._calls: deque[float] = deque()
        self._base_ids: frozenset[int] = frozenset()
        self._base_counts: dict[str, int] = {}
        self._pend_counts: dict[str, int] = {}
        self._pend_since = start
        self._motion_since: float | None = None
        self.dropped = 0

    def step(self, now: float, tracker: SceneTracker, events: list[SceneEvent], *,
             busy: bool = False, frame_changed: bool = False) -> VlmRequest | None:
        ids, counts = tracker.ids(), tracker.counts()
        if counts != self._pend_counts:
            self._pend_counts, self._pend_since = counts, now
        if not frame_changed:
            self._motion_since = None
        elif self._motion_since is None:
            self._motion_since = now
        added = tuple(sorted(ids - self._base_ids))
        union = ids | self._base_ids
        distance = 1 - len(ids & self._base_ids) / len(union) if union else 0.0
        new = tuple(e.track_id for e in events if e.kind == "entered" and e.new_class)

        if new:
            req = VlmRequest("new_class", new)
        elif distance >= self._jaccard:
            req = VlmRequest("scene_change", added)
        elif self._motion_since is not None and now - self._motion_since >= self._motion_hold:
            req = VlmRequest("scene_change")
        elif counts != self._base_counts and now - self._pend_since >= self._count_stable:
            req = VlmRequest("count_change", added)
        elif now - self._last_call >= (self._refresh_person if "person" in counts else self._refresh_empty):
            req = VlmRequest("refresh")
        else:
            return None

        while self._calls and now - self._calls[0] >= 60.0:
            self._calls.popleft()
        if busy or now - self._last_call < self._cooldown or len(self._calls) >= self._max_per_min:
            self.dropped += 1
            return None
        self._last_call = now
        self._calls.append(now)
        self._base_ids, self._base_counts, self._motion_since = ids, counts, None
        return req


def crop_frame(frame: Frame, box: tuple[float, float, float, float], pad: float = 0.2) -> Frame:
    """Crop a normalized xyxy box, padded by `pad` of its size per side. rgb24 only."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    l, r = max(0, int((x1 - px) * frame.width)), min(frame.width, int((x2 + px) * frame.width) + 1)
    t, b = max(0, int((y1 - py) * frame.height)), min(frame.height, int((y2 + py) * frame.height) + 1)
    img = np.frombuffer(frame.data, np.uint8).reshape(frame.height, frame.width, 3)[t:b, l:r]
    return Frame(frame.timestamp, img.tobytes(), img.shape[1], img.shape[0], "rgb24")


class SceneWatcher:
    """Runs detector -> tracker -> gate at `fps`; calls `describe(frame, hint)` when the gate opens."""

    STALE_SECONDS = 10.0

    def __init__(self, *, detector, camera, change, idle: asyncio.Event | None = None, on_update=None,
                 fps: float = 5.0, tracker: SceneTracker | None = None, gate: VlmGate | None = None,
                 clock=time.monotonic) -> None:
        self.detector = detector
        self.describe = None           # async (frame, hint) -> None; set by the pipeline
        self._camera, self._change = camera, change
        self.idle, self._on_update = idle, on_update
        self._interval = 1.0 / fps
        self._clock = clock
        self.tracker = tracker or SceneTracker()
        self._gate = gate or VlmGate(start=clock())
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="detector")
        self._vlm: asyncio.Task | None = None
        self._updated: float | None = None
        self._fps = self._ms = 0.0
        self.frames = self.vlm_calls = 0

    @property
    def available(self) -> bool:
        return self.detector.loaded

    @property
    def metrics(self) -> dict[str, float]:
        return {"detector_frames": self.frames, "detector_fps": round(self._fps, 1),
                "detector_ms": round(self._ms, 1), "vlm_gate_calls": self.vlm_calls,
                "vlm_gate_dropped": self._gate.dropped}

    def summary(self) -> str | None:
        """One-line scene text for the LLM context; None if the detector has gone quiet."""
        now = self._clock()
        if self._updated is None or now - self._updated > self.STALE_SECONDS:
            return None
        return self.tracker.summary(now)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                t0 = self._clock()
                try:
                    frame = await self._camera.capture()
                    if frame is not None:
                        await self._tick(loop, frame)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Detector tick failed")
                await asyncio.sleep(max(0.0, self._interval - (self._clock() - t0)))
        finally:
            if self._vlm is not None:
                self._vlm.cancel()
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def _tick(self, loop, frame: Frame) -> None:
        t = time.perf_counter()
        entities = await loop.run_in_executor(self._pool, self.detector.detect, frame)
        ms = (time.perf_counter() - t) * 1000
        now = self._clock()
        if self._updated is not None:
            self._fps += (1.0 / max(now - self._updated, 1e-3) - self._fps) * 0.2
        self._ms = ms if not self.frames else self._ms + (ms - self._ms) * 0.2
        self._updated, self.frames = now, self.frames + 1
        events = self.tracker.update(now, entities)
        for ev in events:
            logger.info("SCENE %s", ev)
        if self._on_update is not None:
            self._on_update({"fps": round(self._fps, 1), "ms": round(self._ms, 1), "events": [str(e) for e in events],
                             "items": [{"id": e.track_id, "label": e.label, "conf": e.conf, "box": list(e.box)}
                                       for e in entities],
                             "summary": self.tracker.summary(now)})
        busy = (self.idle is not None and not self.idle.is_set()) or (self._vlm is not None and not self._vlm.done())
        req = self._gate.step(now, self.tracker, events, busy=busy, frame_changed=self._change.has_changed(frame))
        if req is not None and self.describe is not None:
            self.vlm_calls += 1
            self._vlm = asyncio.create_task(self._ask(req, frame, entities))

    async def _ask(self, req: VlmRequest, frame: Frame, entities: list[Entity]) -> None:
        by_id = {e.track_id: e for e in entities}
        one = by_id.get(req.track_ids[0]) if len(req.track_ids) == 1 else None
        logger.info("VLM gate: %s%s", req.kind, f" ({one.label})" if one else "")
        try:
            if one is not None and frame.format == "rgb24":
                hint = (f"A detector says this crop shows: {one.label} (confidence {one.conf:.2f}). "
                        "Confirm or correct it; keep the description to one short phrase.")
                await self.describe(crop_frame(frame, one.box), hint)
            else:
                hint = (f"A detector sees: {self.tracker.summary(self._clock())}. "
                        "Use that as a hint and correct it if it is wrong.")
                await self.describe(frame, hint)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("VLM gate request failed")
