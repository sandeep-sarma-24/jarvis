"""Fakes for TurnManager tests: no models, no audio, no network."""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np

from app.attention.engine import AttentionDecision
from app.config import JarvisConfig
from app.context.builder import ContextBuilder
from app.contracts import LatencyRecorder, Transcript
from app.memory.working import WorkingMemory
from app.orchestration.turns import TurnManager
from app.reasoning.model import ReasoningError, ReasoningResult
from app.tts.model import AudioChunk, TTSStreamStats
from app.tts.player import PlaybackOutcome


def transcript(text: str) -> Transcript:
    now = time.time()
    return Transcript(
        text=text, speech_start_time=now - 1.0, speech_end_time=now - 0.3,
        segment_emitted_time=now - 0.1, asr_latency_ms=50.0, audio_seconds=0.7,
    )


class FakeBus:
    def __init__(self):
        self.events = []

    async def publish(self, event):
        self.events.append(event)


class FakeAttention:
    """Attention-positive iff the transcript mentions jarvis."""

    def __init__(self):
        self.working_memory = WorkingMemory(max_size=20)

    def evaluate(self, event):
        ok = "jarvis" in event.transcript.lower()
        self.working_memory.record(event.id, event.transcript, ok)
        return AttentionDecision(should_reason=ok, salience=1.0 if ok else 0.0, reason="t", tier="rule")


class StreamingModel:
    """Yields scripted deltas with a per-delta delay; honours cancel."""

    model_name = "fake-stream"

    def __init__(self, deltas, delay=0.0, fail_after=None, hang_after=None):
        self.deltas = list(deltas)
        self.delay = delay
        self.fail_after = fail_after      # raise ReasoningError after this many deltas
        self.hang_after = hang_after      # block (until cancel) after this many deltas
        self.yielded = 0
        self.finished_at = None
        self.calls = 0

    @property
    def last_stats(self):
        return {}

    def generate(self, messages, *, max_tokens=256, temperature=0.2):
        raise AssertionError("streaming model must be streamed")

    def stream(self, messages, *, max_tokens=256, temperature=0.2, cancel=None):
        self.calls += 1
        for i, d in enumerate(self.deltas):
            if self.fail_after is not None and i == self.fail_after:
                raise ReasoningError("backend died")
            if self.hang_after is not None and i == self.hang_after:
                while not cancel.is_set():
                    time.sleep(0.005)
            if cancel is not None and cancel.is_set():
                return
            if self.delay:
                if cancel is not None and cancel.wait(self.delay):
                    return
            self.yielded += 1
            yield d
        self.finished_at = time.time()

    def warm_up(self):
        pass

    def close(self):
        pass


class PlainModel:
    """Non-streaming model: generate() only."""

    model_name = "fake-plain"

    def __init__(self, text="Hello there, this is a full answer. And a second one.", success=True):
        self.text = text
        self.success = success

    def generate(self, messages, *, max_tokens=256, temperature=0.2):
        return ReasoningResult(text=self.text, model=self.model_name, latency_ms=1.0,
                               success=self.success, error=None if self.success else "boom")


class FakeTTS:
    """stream_to synthesises `chunks` chunks per sentence on a thread."""

    model_name = "fake-tts"
    sample_rate = 16000

    def __init__(self, chunks=2, chunk_delay=0.0, gate: threading.Event | None = None):
        self.chunks = chunks
        self.chunk_delay = chunk_delay
        self.gate = gate      # if set, synthesis waits for it (simulates a stalled TTS)
        self.calls: list[tuple[float, str]] = []

    def load(self):
        pass

    def synthesize(self, text):
        raise NotImplementedError

    def synthesize_stream(self, text):
        raise NotImplementedError

    def close(self):
        pass

    async def stream_to(self, text, sink, *, cancel=None):
        self.calls.append((time.time(), text))

        def run():
            n = 0
            for i in range(self.chunks):
                if self.gate is not None:
                    while not self.gate.wait(0.01):
                        if cancel is not None and cancel.is_set():
                            return n
                if cancel is not None and cancel.is_set():
                    return n
                if self.chunk_delay and cancel is not None and cancel.wait(self.chunk_delay):
                    return n
                chunk = AudioChunk(audio=np.zeros(160, np.float32), sample_rate=16000,
                                   is_last=i == self.chunks - 1)
                object.__setattr__(chunk, "text", text)  # tag for assertions
                if not sink(chunk):
                    return n
                n += 1
            return n

        n = await asyncio.to_thread(run)
        return TTSStreamStats(first_chunk_ms=1.0 if n else None, synth_ms=1.0,
                              audio_seconds=n * 0.01, chunks=n,
                              cancelled=bool(cancel and cancel.is_set()))


class FakePlayer:
    """Records begin/feed/end/stop; on_done fires from another thread after end()."""

    def __init__(self, on_first_audio=None, on_done=None, done_delay=0.02):
        self.on_first_audio = on_first_audio
        self.on_done = on_done
        self.done_delay = done_delay
        self.current: str | None = None
        self.log: list[tuple] = []        # ("begin"|"feed"|"end"|"stop", turn_id, extra)
        self._lock = threading.Lock()
        self._first: set[str] = set()

    def start(self, sample_rate):
        pass

    def begin(self, turn_id):
        with self._lock:
            self.current = turn_id
            self.log.append(("begin", turn_id, time.time()))

    def feed(self, chunk, turn_id):
        with self._lock:
            if turn_id != self.current:
                return False
            self.log.append(("feed", turn_id, getattr(chunk, "text", "")))
            first = turn_id not in self._first
            self._first.add(turn_id)
        if first and self.on_first_audio:
            self.on_first_audio(turn_id)
        return True

    def end(self, turn_id):
        self.log.append(("end", turn_id, time.time()))

        def fire():
            time.sleep(self.done_delay)
            with self._lock:
                current = self.current == turn_id
            if current and self.on_done:
                self.on_done(turn_id, PlaybackOutcome.COMPLETED)

        threading.Thread(target=fire, daemon=True).start()

    def stop(self):
        with self._lock:
            self.current = None
            self.log.append(("stop", None, time.time()))

    def is_playing(self):
        return self.current is not None

    @property
    def stats(self):
        return {}

    def close(self):
        pass

    def fed(self, turn_id=None):
        return [e for e in self.log if e[0] == "feed" and (turn_id is None or e[1] == turn_id)]


class FakeFrontend:
    def __init__(self):
        self.states: list[bool] = []

    def set_playback_active(self, active):
        self.states.append(active)

    @property
    def active(self):
        return bool(self.states) and self.states[-1]


@dataclass
class Msg:
    role: str
    content: str
    source_event_id: str


class FakeConversation:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.messages: list[Msg] = []
        self.assistant_writes: list[tuple[float, str]] = []

    def add_user_message(self, text, source_event_id=None):
        time.sleep(self.delay)
        if not any(m.source_event_id == source_event_id for m in self.messages):
            self.messages.append(Msg("user", text, source_event_id))

    def add_assistant_message(self, text, source_event_id=None):
        self.assistant_writes.append((time.time(), text))
        self.messages.append(Msg("assistant", text, source_event_id))

    def recent_messages(self, limit=20):
        time.sleep(self.delay)
        return list(self.messages[-limit:])


class FakeMemory:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.conversation = FakeConversation(delay)
        self.retrieved: list[str] = []

    def retrieve_context(self, query, *, limit):
        time.sleep(self.delay)
        self.retrieved.append(query)
        return ["[memory 2026-01-01] likes tea"]

    def maybe_remember(self, text, source_event_id):
        time.sleep(self.delay)
        return None


class FakeCamera:
    def __init__(self):
        self.captures = 0

    async def capture(self):
        self.captures += 1
        return object()


class FakeAnalyzer:
    def __init__(self, delay=0.0, description="a desk with a laptop"):
        self.delay = delay
        self.description = description

    async def analyze(self, frame):
        await asyncio.sleep(self.delay)

        class A:
            description = self.description
        return A()


def make_manager(*, model=None, tts=None, speech=True, memory=None, vision=None,
                 frontend=None, cfg=None, done_delay=0.02):
    cfg = cfg or JarvisConfig()
    recorder = LatencyRecorder()
    holder: dict[str, TurnManager] = {}
    player = FakePlayer(
        on_first_audio=lambda tid: holder["tm"].on_first_audio(tid),
        on_done=lambda tid, o: holder["tm"].on_playback_done(tid, o),
        done_delay=done_delay,
    ) if speech else None
    frontend = frontend or FakeFrontend()
    tm = TurnManager(
        cfg=cfg, bus=FakeBus(), attention=FakeAttention(),
        context_builder=ContextBuilder(), model=model or StreamingModel(["Hi there, friend. ", "How are you today? "]),
        tts=(tts or FakeTTS()) if speech else None, player=player, memory=memory,
        frontend=frontend, recorder=recorder, on_demand_vision=vision,
        session_id="s",
    )
    holder["tm"] = tm
    return tm, player, frontend, recorder


async def wait_for(cond, timeout=2.0):
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)
