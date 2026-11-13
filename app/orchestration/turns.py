"""TurnManager — one Turn per attention-positive utterance, latest wins.

Transcript → SpeechEvent + attention → [persist ‖ history ‖ memory ‖ vision]
→ context → streaming reasoning → sentence splitter → bounded queue → TTS → player.

Every blocking call runs off the event loop. `turn.cancel` is the single stop
signal; a newer transcript, a barge-in or shutdown sets it.
See docs/architecture/06-engineering-contracts.md, section 6.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from app.contracts import STAGES, LatencyRecorder, Transcript, Turn
from app.events.types import Modality, ResponseEvent, SpeechEvent
from app.orchestration.sentences import SentenceSplitter
from app.reasoning.model import StreamingReasoningModel
from app.response.decision import decide

logger = logging.getLogger("jarvis.turns")

DELTA_QUEUE_SIZE = 32     # LLM deltas between the worker thread and the loop
SENTENCE_QUEUE_SIZE = 8   # sentences waiting for TTS

_VISION_RE = re.compile(
    r"\b(see|seeing|look|looking|watch|watching|visual|camera|showing)\b", re.IGNORECASE
)


def is_vision_request(text: str) -> bool:
    return _VISION_RE.search(text) is not None


def format_multimodal_context(snap) -> list[str] | None:
    """Format a fusion snapshot into context strings for ContextBuilder.

    The final visual observation is marked as current to help the reasoning
    model prioritize it over older (potentially stale) descriptions.
    """
    parts = [f"[audio] {obs}" for obs in snap.audio_observations]
    visuals = snap.visual_observations
    parts += [f"[visual] {obs}" for obs in visuals[:-1]]
    if visuals:
        parts.append(f"[visual — current] {visuals[-1]}")
    return parts or None


@dataclass
class _TurnState:
    turn: Turn
    event: SpeechEvent
    task: asyncio.Task | None = None
    text: str = ""                                    # visible model output
    spoken: list[str] = field(default_factory=list)   # sentences handed to TTS
    playback_started: bool = False
    playback_done: asyncio.Future | None = None
    tts_failed: bool = False
    recorded: bool = False


class TurnManager:
    def __init__(
        self,
        *,
        cfg,
        bus,
        attention,
        context_builder,
        model,
        tts=None,
        player=None,
        memory=None,
        fusion=None,
        frontend=None,
        recorder: LatencyRecorder,
        on_demand_vision=None,       # (camera, analyzer) or None
        session_id: str = "",
        visual_context_ring=None,    # deque[(timestamp, description)] fallback for fusion
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._attention = attention
        self._context_builder = context_builder
        self._model = model
        self._tts = tts
        self._player = player
        self._memory = memory
        self._fusion = fusion
        self._frontend = frontend
        self._recorder = recorder
        self._vision = on_demand_vision
        self._session_id = session_id
        self._ring = visual_context_ring
        self._speech = (
            tts is not None and player is not None and cfg.tts.enabled and cfg.playback.enabled
        )
        self._active: _TurnState | None = None
        self._last_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Cleared while a voice turn runs; ambient vision waits on it.
        self.idle = asyncio.Event()
        self.idle.set()

    # ── Entry points ───────────────────────────────────────────────────

    async def handle_transcript(self, transcript: Transcript) -> None:
        self._loop = asyncio.get_running_loop()
        received = time.time()
        event = SpeechEvent(
            session_id=self._session_id,
            correlation_id=uuid.uuid4().hex,
            source=Modality.AUDIO,
            confidence=0.0,
            transcript=transcript.text,
            duration_seconds=transcript.audio_seconds,
            speech_end_timestamp=transcript.speech_end_time,
            metadata={
                **transcript.metadata,
                "asr_latency_ms": round(transcript.asr_latency_ms, 1),
                "total_speech_to_event_ms": round((received - transcript.speech_end_time) * 1000, 1),
            },
        )
        await self._bus.publish(event)

        if self._attention is None:
            return
        decision = self._attention.evaluate(event)
        if not decision.should_reason:
            print(
                f"\n[ATTENTION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  should_reason=false\n"
                f"\n[NO RESPONSE]\n"
            )
            return
        print(
            f"\n[ATTENTION]\n"
            f"  event_id={event.id[:12]}\n"
            f"  should_reason=true\n"
            f"  salience={decision.salience:.2f}\n"
        )
        if not (self._cfg.reasoning.enabled and self._cfg.response.enabled):
            return

        self._cancel_active()
        turn = Turn(turn_id=event.correlation_id, speech_end_time=transcript.speech_end_time)
        turn.mark("segment_emitted", transcript.segment_emitted_time)
        turn.mark("asr_done", received)
        turn.mark("attention_done")
        st = _TurnState(turn=turn, event=event)
        prev = self._last_task
        self._active = st
        if self._cfg.perception.pause_during_turn:
            self.idle.clear()
        st.task = asyncio.create_task(self._run(st, prev))
        # Covers a task cancelled before it ever ran; a no-op otherwise.
        st.task.add_done_callback(lambda _t: self._record(st, "cancelled"))
        self._last_task = st.task

    def on_barge_in(self, t: float) -> None:
        if self._active is not None:
            logger.info("BARGE_IN turn=%s", self._active.turn.turn_id[:12])
        self._cancel_active()

    async def shutdown(self) -> None:
        self._cancel_active()
        task = self._last_task
        if task is not None and not task.done():
            await asyncio.wait({task}, timeout=2.0)
        if self._player is not None:
            self._player.stop()

    # Player callbacks: called on the player's thread.
    def on_first_audio(self, turn_id: str) -> None:
        self._call_on_loop(self._first_audio, turn_id)

    def on_playback_done(self, turn_id: str, outcome) -> None:
        self._call_on_loop(self._playback_finished, turn_id, outcome)

    # ── Cancellation ───────────────────────────────────────────────────

    def _cancel_active(self) -> None:
        st = self._active
        if st is None:
            return
        self._stop(st)
        if st.task is not None:
            st.task.cancel()

    def _stop(self, st: _TurnState) -> None:
        if self._active is st:
            self._active = None
        st.turn.cancel.set()
        if self._player is not None:
            self._player.stop()
        if st.playback_started and self._frontend is not None:
            self._frontend.set_playback_active(False)
        self.idle.set()

    # ── Turn task ──────────────────────────────────────────────────────

    async def _run(self, st: _TurnState, prev: asyncio.Task | None) -> None:
        turn = st.turn
        try:
            if prev is not None and not prev.done():
                # Let the superseded turn persist what it said so history matches.
                await asyncio.wait({prev}, timeout=0.5)
            messages = await self._prepare(st)
            turn.mark("context_ready")
            outcome = await self._respond(st, messages)
        except asyncio.CancelledError:
            outcome = "cancelled"
        except Exception:
            logger.exception("Turn %s failed", turn.turn_id[:12])
            outcome = "failed"
            self._stop(st)
        await self._finish(st, outcome)

    async def _prepare(self, st: _TurnState) -> list[dict[str, str]]:
        cfg = self._cfg
        event = st.event
        text = event.transcript
        conversation = self._memory.conversation if self._memory is not None else None
        use_memory = self._memory is not None and cfg.memory.enabled

        jobs: dict[str, object] = {}
        if conversation is not None:
            jobs["persist"] = asyncio.to_thread(conversation.add_user_message, text, event.id)
            if cfg.memory.enabled:
                jobs["history"] = asyncio.to_thread(
                    conversation.recent_messages, cfg.memory.max_recent_messages
                )
        if use_memory:
            jobs["memories"] = asyncio.to_thread(
                self._memory.retrieve_context, text, limit=cfg.memory.max_retrieved_memories
            )
            jobs["remember"] = asyncio.to_thread(self._memory.maybe_remember, text, event.id)
        if self._vision is not None and is_vision_request(text):
            jobs["vision"] = self._live_vision()

        results = dict(zip(jobs, await asyncio.gather(*jobs.values(), return_exceptions=True)))
        for name, r in results.items():
            if isinstance(r, BaseException):
                logger.error("Turn %s: %s failed: %r", st.turn.turn_id[:12], name, r)
                results[name] = None

        conv_messages = None
        if results.get("history") is not None:
            conv_messages = [
                {"role": m.role, "content": m.content}
                for m in results["history"]
                if m.source_event_id != event.id
            ]
        mem_context = results.get("memories") or None

        multimodal_ctx: list[str] | None = None
        vis_ctx: list[str] | None = None
        if results.get("vision"):
            multimodal_ctx = [f"[visual — live] {results['vision']}"]
        elif is_vision_request(text):
            # Without this the system prompt's "you have a camera" makes the model invent a view.
            multimodal_ctx = ["[visual — live] unavailable: no camera image right now. "
                              "If asked what you can see, say you cannot see at the moment."]
        if self._fusion is not None and self._fusion.enabled:
            snap = self._fusion.snapshot(exclude_event_ids={event.id})
            fusion_ctx = format_multimodal_context(snap)
            if fusion_ctx:
                multimodal_ctx = (multimodal_ctx or []) + fusion_ctx
            # The snapshot may lack visuals while an analysis is still in flight;
            # the ring holds attention-worthy visual events as they complete.
            if not snap.visual_observations and self._ring:
                multimodal_ctx = (multimodal_ctx or []) + [f"[visual] {d}" for _, d in self._ring]
            logger.info(
                "FUSION_SNAPSHOT audio=%d visual=%d chars=%d",
                len(snap.audio_observations), len(snap.visual_observations), snap.total_characters,
            )
        elif self._ring:
            vis_ctx = [d for _, d in self._ring]

        ctx = self._context_builder.build(
            event, self._attention.working_memory,
            conversation_messages=conv_messages,
            memory_context=mem_context,
            visual_context=vis_ctx,
            multimodal_context=multimodal_ctx,
        )
        messages = self._context_builder.to_messages(ctx)
        logger.debug("Context: %d messages, %d chars", len(messages), ctx.total_characters)
        return messages

    async def _live_vision(self) -> str | None:
        camera, analyzer = self._vision

        async def look() -> str | None:
            frame = await camera.capture()
            if frame is None:
                return None
            analysis = await analyzer.analyze(frame)
            return None if analysis is None else analysis.description

        timeout = self._cfg.perception.on_demand_timeout_seconds
        try:
            desc = await asyncio.wait_for(look(), timeout)
        except TimeoutError:
            logger.warning("ON_DEMAND_VISION timed out after %.1fs", timeout)
            return None
        if desc:
            logger.info("ON_DEMAND_VISION desc=%r", desc)
        return desc

    async def _respond(self, st: _TurnState, messages: list[dict[str, str]]) -> str:
        cfg = self._cfg
        turn = st.turn
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=SENTENCE_QUEUE_SIZE)
        speaker = asyncio.create_task(self._speak(st, sentences)) if self._speech else None
        splitter = SentenceSplitter()
        budget = cfg.tts.max_text_length
        error: str | None = None
        thinking: str | None = None
        reason: str | None = None
        emitted = False

        async def emit(batch: list[str]) -> bool:
            """Queue sentences for speech; False once max_text_length is reached."""
            nonlocal budget, emitted
            for sentence in batch:
                if len(sentence) > budget:
                    if emitted or budget <= 0:
                        budget = 0
                        return False
                    sentence = sentence[:budget]
                budget -= len(sentence)
                emitted = True
                if speaker is None:
                    st.spoken.append(sentence)
                else:
                    await sentences.put(sentence)
            return budget > 0

        try:
            try:
                more = True
                if isinstance(self._model, StreamingReasoningModel):
                    async with contextlib.aclosing(self._stream_deltas(turn, messages)) as deltas:
                        async for delta in deltas:
                            turn.mark("llm_first_token")
                            st.text += delta
                            if not await emit(splitter.feed(delta)):
                                more = False
                                break
                else:
                    result = await asyncio.to_thread(
                        self._model.generate, messages,
                        max_tokens=cfg.reasoning.max_output_tokens,
                        temperature=cfg.reasoning.temperature,
                    )
                    resp = decide(st.event.id, result)
                    thinking, reason = resp.thinking, resp.reason
                    if not result.success:
                        error = result.error or "unknown error"
                    if resp.respond and resp.text:
                        turn.mark("llm_first_token")
                        st.text = resp.text
                        more = await emit(splitter.feed(resp.text))
                turn.mark("llm_done")
                if more and (tail := splitter.flush()):
                    await emit([tail])
            except Exception as exc:  # reasoning failed: speak only complete sentences already out
                error = str(exc) or type(exc).__name__
                logger.error("Turn %s reasoning failed: %s", turn.turn_id[:12], error)
            if speaker is not None:
                await sentences.put(None)
        except BaseException:
            if speaker is not None:
                speaker.cancel()
            raise

        text = st.text.strip()
        llm_ms = 0.0
        if "llm_done" in turn.marks and "context_ready" in turn.marks:
            llm_ms = (turn.marks["llm_done"] - turn.marks["context_ready"]) * 1000
        if text:
            thinking_block = f"\n[THINKING]\n  {thinking}\n" if thinking else ""
            print(
                f"\n[REASONING]\n"
                f"  model={self._model.model_name}\n"
                f"  latency={llm_ms:.0f}ms\n"
                f"{thinking_block}"
                f"\n[RESPONSE]\n"
                f"  {text}\n"
            )
            await self._bus.publish(ResponseEvent(
                session_id=st.event.session_id,
                correlation_id=turn.turn_id,
                source=Modality.SYSTEM,
                event_id=st.event.id,
                text=text,
                reasoning_latency_ms=round(llm_ms, 1),
                speech_end_timestamp=turn.speech_end_time,
            ))
        else:
            if reason is None:
                reason = f"reasoning failed: {error}" if error else "reasoning produced empty response"
            print(
                f"\n[RESPONSE_DECISION]\n"
                f"  event_id={st.event.id[:12]}\n"
                f"  respond=false\n"
                f"  reason=\"{reason}\"\n"
            )
        logger.info(
            "REASONING turn=%s model=%s latency_ms=%.0f success=%s chars=%d",
            turn.turn_id[:12], self._model.model_name, llm_ms, error is None, len(text),
        )

        if speaker is not None:
            await speaker
            if st.playback_started and st.playback_done is not None:
                await st.playback_done

        if turn.cancelled:
            return "cancelled"
        if error or st.tts_failed:
            return "failed"
        return "completed" if text else "no_response"

    async def _stream_deltas(self, turn: Turn, messages) -> AsyncIterator[str]:
        """Drive model.stream() on a worker thread; deltas cross a bounded queue."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=DELTA_QUEUE_SIZE)
        stop = threading.Event()  # consumer went away (closed early)

        def put(item) -> bool:
            """Blocks while the queue is full; False if the turn was cancelled."""
            try:
                fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
            except RuntimeError:  # loop closed
                return False
            while True:
                try:
                    fut.result(timeout=0.05)
                    return True
                except TimeoutError:
                    if turn.cancel.is_set() or stop.is_set():
                        fut.cancel()
                        return False
                except Exception:
                    return False

        def work() -> None:
            gen = None
            try:
                gen = self._model.stream(
                    messages,
                    max_tokens=self._cfg.reasoning.max_output_tokens,
                    temperature=self._cfg.reasoning.temperature,
                    cancel=turn.cancel,
                )
                for delta in gen:
                    if delta and not put(("delta", delta)):
                        return
                put(("end", None))
            except Exception as exc:
                put(("error", exc))
            finally:
                if gen is not None:
                    gen.close()

        loop.run_in_executor(None, work)
        try:
            while True:
                kind, value = await q.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise value  # type: ignore[misc]
                yield value  # type: ignore[misc]
        finally:
            stop.set()

    async def _speak(self, st: _TurnState, q: asyncio.Queue[str | None]) -> None:
        turn = st.turn
        tid = turn.turn_id

        def sink(chunk) -> bool:
            turn.mark("tts_first_chunk")
            return self._player.feed(chunk, turn_id=tid)

        try:
            # Always drain to the sentinel so the producer never blocks on a full queue.
            while (sentence := await q.get()) is not None:
                if turn.cancelled or st.tts_failed:
                    continue
                try:
                    if not st.playback_started:
                        st.playback_done = asyncio.get_running_loop().create_future()
                        self._player.begin(tid)
                        st.playback_started = True
                        if self._frontend is not None:
                            self._frontend.set_playback_active(True)
                    turn.mark("first_sentence")
                    st.spoken.append(sentence)
                    await self._tts.stream_to(sentence, sink, cancel=turn.cancel)
                except Exception:
                    logger.exception("Turn %s TTS failed", tid[:12])
                    st.tts_failed = True
        finally:
            if st.playback_started and not turn.cancelled:
                self._player.end(tid)

    # ── Completion ─────────────────────────────────────────────────────

    async def _finish(self, st: _TurnState, outcome: str) -> None:
        if self._active is st:
            self._active = None
        if self._active is None:
            self.idle.set()
        conversation = self._memory.conversation if self._memory is not None else None
        spoken = " ".join(st.spoken)
        if spoken and conversation is not None:
            try:
                await asyncio.to_thread(
                    conversation.add_assistant_message, spoken, st.event.id + "_response"
                )
            except Exception:
                logger.exception("Turn %s: persisting assistant message failed", st.turn.turn_id[:12])
        self._record(st, outcome)

    def _record(self, st: _TurnState, outcome: str) -> None:
        if st.recorded:
            return
        st.recorded = True
        row = self._recorder.record(st.turn, outcome)
        stages = " ".join(f"{k}={row[k]:.0f}" for k in STAGES if k in row)
        logger.info("TURN turn=%s outcome=%s %s", st.turn.turn_id[:12], outcome, stages)

    def _call_on_loop(self, fn, *args) -> None:
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:  # loop already closed during shutdown
            pass

    def _first_audio(self, turn_id: str) -> None:
        st = self._active
        if st is None or st.turn.turn_id != turn_id:
            return
        st.turn.mark("first_audio")
        print(
            f"\n[TTS]\n"
            f"  event_id={st.event.id[:12]}\n"
            f"  speech_end_to_audio={st.turn.elapsed_ms('first_audio'):.0f}ms\n"
        )

    def _playback_finished(self, turn_id: str, outcome) -> None:
        st = self._active
        # Stale callbacks (an older, superseded turn) must not touch the mic state.
        if st is None or st.turn.turn_id != turn_id:
            return
        st.turn.mark("playback_done")
        if self._frontend is not None:
            self._frontend.set_playback_active(False)
        if st.playback_done is not None and not st.playback_done.done():
            st.playback_done.set_result(outcome)
