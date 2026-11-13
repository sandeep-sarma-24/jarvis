"""TurnManager behaviour with fake model, TTS, player, memory and frontend."""
from __future__ import annotations

import asyncio
import threading
import time


from app.config import JarvisConfig
from app.contracts import STAGES
from app.events.types import ResponseEvent, SpeechEvent
from app.orchestration import turns as turns_mod
from app.orchestration.turns import is_vision_request
from app.tts.player import PlaybackOutcome

from tests.test_orchestration.fakes import (
    FakeAnalyzer, FakeCamera, FakeMemory, FakeTTS, PlainModel, StreamingModel,
    make_manager, transcript, wait_for,
)


async def finish(tm):
    await asyncio.wait_for(asyncio.shield(tm._last_task), 3.0)


def outcomes(recorder):
    return [r["outcome"] for r in recorder.turns]


# ── Streaming and ordering ─────────────────────────────────────────────

async def test_first_sentence_reaches_tts_before_model_finishes():
    model = StreamingModel(["The first sentence is here. ", "Then ", "more ", "words ", "follow. "], delay=0.1)
    tts = FakeTTS()
    tm, player, _, recorder = make_manager(model=model, tts=tts)
    await tm.handle_transcript(transcript("jarvis tell me something"))
    await finish(tm)
    assert tts.calls[0][1] == "The first sentence is here."
    assert tts.calls[0][0] < model.finished_at
    assert outcomes(recorder) == ["completed"]


async def test_sentences_spoken_in_order_and_fed_to_player_in_order():
    deltas = ["Alpha sentence one. ", "Beta sentence two. ", "Gamma sentence three."]
    tts = FakeTTS(chunks=3)
    tm, player, _, _ = make_manager(model=StreamingModel(deltas), tts=tts)
    await tm.handle_transcript(transcript("jarvis go"))
    await finish(tm)
    spoken = [t for _, t in tts.calls]
    assert spoken == ["Alpha sentence one.", "Beta sentence two.", "Gamma sentence three."]
    fed = [e[2] for e in player.fed()]
    assert fed == [s for s in spoken for _ in range(3)]
    kinds = [e[0] for e in player.log]
    assert kinds[0] == "begin" and kinds[-1] == "end"


async def test_handle_transcript_returns_without_waiting_for_turn():
    tm, *_ = make_manager(model=StreamingModel(["Slow answer here. "], delay=0.5))
    t0 = time.perf_counter()
    await tm.handle_transcript(transcript("jarvis hello"))
    assert time.perf_counter() - t0 < 0.05
    assert not tm._last_task.done()
    await tm.shutdown()


async def test_speech_event_published_with_capture_speech_end():
    tm, *_ = make_manager()
    t = transcript("jarvis hi")
    await tm.handle_transcript(t)
    await finish(tm)
    speech = [e for e in tm._bus.events if isinstance(e, SpeechEvent)]
    resp = [e for e in tm._bus.events if isinstance(e, ResponseEvent)]
    assert speech[0].speech_end_timestamp == t.speech_end_time
    assert speech[0].metadata["asr_latency_ms"] == 50.0
    assert len(resp) == 1
    assert resp[0].correlation_id == speech[0].correlation_id
    assert resp[0].text == "Hi there, friend. How are you today?"
    assert resp[0].speech_end_timestamp == t.speech_end_time


# ── Cancellation ───────────────────────────────────────────────────────

async def test_superseding_transcript_cancels_and_no_old_audio_after_stop():
    slow = StreamingModel([f"Old sentence number {i}. " for i in range(50)], delay=0.03)
    tts = FakeTTS(chunks=5, chunk_delay=0.01)
    tm, player, _, recorder = make_manager(model=slow, tts=tts)
    await tm.handle_transcript(transcript("jarvis tell a long story"))
    first = tm._active.turn.turn_id
    await wait_for(lambda: player.fed(first))

    tm._model = StreamingModel(["New answer here. "])
    await tm.handle_transcript(transcript("jarvis stop, new question"))
    second = tm._active.turn.turn_id
    stop_index = next(i for i, e in enumerate(player.log) if e[0] == "stop")
    await finish(tm)

    assert not any(e[0] == "feed" and e[1] == first for e in player.log[stop_index:])
    assert player.fed(second)
    assert outcomes(recorder) == ["cancelled", "completed"]


async def test_cancellation_completes_within_200ms():
    model = StreamingModel([f"Sentence number {i} is here. " for i in range(100)], delay=0.05)
    tts = FakeTTS(chunks=20, chunk_delay=0.05)
    tm, player, _, recorder = make_manager(model=model, tts=tts)
    await tm.handle_transcript(transcript("jarvis talk"))
    task = tm._last_task
    await wait_for(lambda: player.fed())
    t0 = time.perf_counter()
    tm.on_barge_in(time.time())
    await asyncio.wait_for(asyncio.shield(task), 1.0)
    assert time.perf_counter() - t0 < 0.2
    assert outcomes(recorder) == ["cancelled"]


async def test_non_attention_transcript_does_not_cancel():
    tm, player, _, recorder = make_manager(model=StreamingModel(["Answer one here. "], delay=0.1))
    await tm.handle_transcript(transcript("jarvis question"))
    task = tm._last_task
    await tm.handle_transcript(transcript("the cat sat on the mat"))
    assert tm._last_task is task
    assert not tm._active.turn.cancelled
    assert not any(e[0] == "stop" for e in player.log)
    await finish(tm)
    assert outcomes(recorder) == ["completed"]


async def test_barge_in_cancels_and_releases_mic():
    model = StreamingModel([f"Sentence {i} goes here. " for i in range(50)], delay=0.03)
    tm, player, frontend, recorder = make_manager(model=model)
    await tm.handle_transcript(transcript("jarvis go on"))
    await wait_for(lambda: player.fed())
    assert frontend.active
    tm.on_barge_in(time.time())
    await finish(tm)
    assert any(e[0] == "stop" for e in player.log)
    assert not frontend.active
    assert outcomes(recorder) == ["cancelled"]


async def test_stale_on_done_does_not_unsuppress_next_turn():
    model = StreamingModel([f"Sentence {i} goes here. " for i in range(50)], delay=0.03)
    tm, player, frontend, _ = make_manager(model=model)
    await tm.handle_transcript(transcript("jarvis first"))
    first = tm._active.turn.turn_id
    await wait_for(lambda: player.fed(first))

    tm._model = StreamingModel([f"Second {i} sentence here. " for i in range(50)], delay=0.03)
    await tm.handle_transcript(transcript("jarvis second"))
    second = tm._active.turn.turn_id
    await wait_for(lambda: player.fed(second))
    assert frontend.active

    # The old turn's playback reports done while the new turn is playing.
    tm.on_playback_done(first, PlaybackOutcome.INTERRUPTED)
    await asyncio.sleep(0.02)
    assert frontend.active

    tm.on_playback_done(second, PlaybackOutcome.COMPLETED)
    await asyncio.sleep(0.02)
    assert not frontend.active
    await tm.shutdown()


async def test_shutdown_with_turn_in_flight_returns_promptly():
    model = StreamingModel(["Never ending. "], hang_after=0)
    tm, player, _, recorder = make_manager(model=model)
    await tm.handle_transcript(transcript("jarvis hang"))
    await asyncio.sleep(0.05)
    t0 = time.perf_counter()
    await tm.shutdown()
    assert time.perf_counter() - t0 < 0.3
    assert tm._last_task.done()
    assert outcomes(recorder) == ["cancelled"]


async def test_task_cancelled_before_start_is_still_recorded():
    tm, player, _, recorder = make_manager()
    await tm.handle_transcript(transcript("jarvis one"))
    await tm.handle_transcript(transcript("jarvis two"))  # cancels one before it ran
    await finish(tm)
    await asyncio.sleep(0)
    assert sorted(outcomes(recorder)) == ["cancelled", "completed"]


# ── Backpressure ───────────────────────────────────────────────────────

async def test_bounded_queues_apply_backpressure_and_lose_nothing(monkeypatch):
    monkeypatch.setattr(turns_mod, "DELTA_QUEUE_SIZE", 2)
    monkeypatch.setattr(turns_mod, "SENTENCE_QUEUE_SIZE", 2)
    words = [f"Sentence number {i} ends here. " for i in range(40)]
    model = StreamingModel(words)
    gate = threading.Event()
    tts = FakeTTS(chunks=1, gate=gate)
    tm, player, _, recorder = make_manager(model=model, tts=tts)
    await tm.handle_transcript(transcript("jarvis count"))
    await asyncio.sleep(0.2)
    # TTS is stalled on sentence 1: queued sentences + queued deltas + in-flight are all
    # that the model may have produced.
    assert model.yielded <= 1 + 2 + 2 + 3
    gate.set()
    await finish(tm)
    assert [t for _, t in tts.calls] == [w.strip() for w in words]
    assert outcomes(recorder) == ["completed"]


# ── Context gathering ──────────────────────────────────────────────────

async def test_history_retrieval_and_vision_run_concurrently():
    memory = FakeMemory(delay=0.15)
    vision = (FakeCamera(), FakeAnalyzer(delay=0.15))
    tm, *_ , recorder = make_manager(memory=memory, vision=vision)
    await tm.handle_transcript(transcript("jarvis what do you see"))
    await finish(tm)
    row = recorder.turns[0]
    gather_ms = row["context_ready"] - row["attention_done"]
    assert gather_ms < 280, gather_ms   # sum would be ~600ms
    assert vision[0].captures == 1
    assert memory.retrieved == ["jarvis what do you see"]


def test_vision_keyword_matches_whole_words_only():
    assert is_vision_request("what do you see")
    assert is_vision_request("Look at this")
    assert is_vision_request("what am I looking at")
    assert not is_vision_request("it seems fine")
    assert not is_vision_request("overlooked details")


async def test_vision_not_run_for_seems():
    vision = (FakeCamera(), FakeAnalyzer())
    tm, *_ = make_manager(vision=vision)
    await tm.handle_transcript(transcript("jarvis it seems cold"))
    await finish(tm)
    assert vision[0].captures == 0


async def test_on_demand_vision_is_bounded_by_timeout():
    cfg = JarvisConfig()
    cfg.perception.on_demand_timeout_seconds = 0.05
    vision = (FakeCamera(), FakeAnalyzer(delay=5.0))
    tm, *_, recorder = make_manager(vision=vision, cfg=cfg)
    t0 = time.perf_counter()
    await tm.handle_transcript(transcript("jarvis look here"))
    await finish(tm)
    assert time.perf_counter() - t0 < 1.0
    assert outcomes(recorder) == ["completed"]


async def test_history_excludes_current_utterance_and_memories_reach_model():
    memory = FakeMemory()
    captured = {}

    class Spy(StreamingModel):
        def stream(self, messages, **kw):
            captured["messages"] = messages
            return super().stream(messages, **kw)

    tm, *_ = make_manager(memory=memory, model=Spy(["Fine answer here. "]))
    await tm.handle_transcript(transcript("jarvis remember tea"))
    await finish(tm)
    contents = [m["content"] for m in captured["messages"]]
    assert contents.count("jarvis remember tea") == 1
    assert any("likes tea" in c for c in contents)


# ── Fallback and failures ──────────────────────────────────────────────

async def test_non_streaming_model_fallback():
    tts = FakeTTS()
    tm, player, _, recorder = make_manager(model=PlainModel(), tts=tts)
    await tm.handle_transcript(transcript("jarvis hello"))
    await finish(tm)
    assert [t for _, t in tts.calls] == ["Hello there, this is a full answer.", "And a second one."]
    assert outcomes(recorder) == ["completed"]


async def test_non_streaming_failure_uses_decision_policy():
    tts = FakeTTS()
    tm, player, _, recorder = make_manager(model=PlainModel(success=False), tts=tts)
    await tm.handle_transcript(transcript("jarvis hello"))
    await finish(tm)
    assert tts.calls == []
    assert outcomes(recorder) == ["failed"]


async def test_non_streaming_think_block_never_spoken():
    tts = FakeTTS()
    model = PlainModel(text="<think>hmm let me think</think>The answer is four.")
    tm, *_ = make_manager(model=model, tts=tts)
    await tm.handle_transcript(transcript("jarvis two plus two"))
    await finish(tm)
    assert [t for _, t in tts.calls] == ["The answer is four."]


async def test_reasoning_failure_before_first_sentence_means_no_speech():
    tts = FakeTTS()
    tm, player, frontend, recorder = make_manager(model=StreamingModel(["x"], fail_after=0), tts=tts)
    await tm.handle_transcript(transcript("jarvis hello"))
    await finish(tm)
    assert tts.calls == []
    assert not any(e[0] == "begin" for e in player.log)
    assert frontend.states == []
    assert outcomes(recorder) == ["failed"]
    assert not [e for e in tm._bus.events if isinstance(e, ResponseEvent)]


async def test_mid_stream_failure_speaks_partial_text():
    memory = FakeMemory()
    tts = FakeTTS()
    model = StreamingModel(["One full sentence here. ", "Two partial", " more"], fail_after=2)
    tm, player, frontend, recorder = make_manager(model=model, tts=tts, memory=memory)
    await tm.handle_transcript(transcript("jarvis hello"))
    await finish(tm)
    assert [t for _, t in tts.calls] == ["One full sentence here."]
    assert outcomes(recorder) == ["failed"]
    assert not frontend.active
    assert [t for _, t in memory.conversation.assistant_writes] == ["One full sentence here."]


async def test_tts_disabled_still_completes_turn():
    memory = FakeMemory()
    tm, player, frontend, recorder = make_manager(speech=False, memory=memory)
    await tm.handle_transcript(transcript("jarvis hi"))
    await finish(tm)
    assert outcomes(recorder) == ["completed"]
    assert frontend.states == []
    assert [t for _, t in memory.conversation.assistant_writes] == ["Hi there, friend. How are you today?"]


async def test_max_text_length_enforced():
    cfg = JarvisConfig()
    cfg.tts.max_text_length = 45
    tts = FakeTTS()
    model = StreamingModel(["First sentence is quite long. ", "Second one too. ", "Third. "])
    tm, *_ = make_manager(model=model, tts=tts, cfg=cfg)
    await tm.handle_transcript(transcript("jarvis go"))
    await finish(tm)
    spoken = [t for _, t in tts.calls]
    assert spoken == ["First sentence is quite long.", "Second one too."]
    assert sum(len(s) for s in spoken) <= 45


# ── Persistence and metrics ────────────────────────────────────────────

async def test_assistant_message_persisted_after_speech_start():
    memory = FakeMemory()
    tm, *_, recorder = make_manager(memory=memory)
    await tm.handle_transcript(transcript("jarvis hi"))
    turn = tm._active.turn
    await finish(tm)
    (written_at, text), = memory.conversation.assistant_writes
    assert written_at >= turn.marks["first_audio"]
    assert text == "Hi there, friend. How are you today?"
    roles = [m.role for m in memory.conversation.messages]
    assert roles == ["user", "assistant"]


async def test_cancel_persists_only_spoken_text():
    memory = FakeMemory()
    model = StreamingModel(["Spoken sentence one. ", "Never spoken. "], hang_after=1)
    tm, player, *_ = make_manager(model=model, memory=memory)
    await tm.handle_transcript(transcript("jarvis hi"))
    await wait_for(lambda: player.fed())
    tm.on_barge_in(time.time())
    await finish(tm)
    assert [t for _, t in memory.conversation.assistant_writes] == ["Spoken sentence one."]


async def test_latency_marks_present_monotonic_and_recorded():
    tm, *_, recorder = make_manager()
    for i in range(3):
        await tm.handle_transcript(transcript(f"jarvis question {i}"))
        turn = tm._active.turn
        await finish(tm)
        ordered = ["speech_end", "segment_emitted", "asr_done", "attention_done", "context_ready",
                   "llm_first_token", "first_sentence", "tts_first_chunk", "first_audio", "playback_done"]
        assert set(STAGES) <= set(turn.marks), set(STAGES) - set(turn.marks)
        ts = [turn.marks[s] for s in ordered]
        assert ts == sorted(ts)
        assert turn.marks["llm_done"] >= turn.marks["llm_first_token"]
    assert outcomes(recorder) == ["completed"] * 3
    assert recorder.summary()["first_audio"]["n"] == 3


async def test_idle_event_cleared_during_turn():
    tm, *_ = make_manager(model=StreamingModel(["Slow one here. "], delay=0.1))
    assert tm.idle.is_set()
    await tm.handle_transcript(transcript("jarvis hi"))
    assert not tm.idle.is_set()
    await finish(tm)
    assert tm.idle.is_set()


async def test_no_response_outcome_for_empty_answer():
    tm, player, _, recorder = make_manager(model=StreamingModel([]))
    await tm.handle_transcript(transcript("jarvis hi"))
    await finish(tm)
    assert outcomes(recorder) == ["no_response"]
    assert not player.log


async def test_vision_request_without_a_camera_image_tells_the_model_it_cannot_see():
    captured = {}

    class Spy(StreamingModel):
        def stream(self, messages, **kw):
            captured["messages"] = messages
            return super().stream(messages, **kw)

    tm, *_ = make_manager(model=Spy(["I cannot see right now. "]))  # no on-demand vision wired
    await tm.handle_transcript(transcript("jarvis what do you see"))
    await finish(tm)
    assert "[visual — live] unavailable" in "\n".join(m["content"] for m in captured["messages"])
