# 06 — Engineering Contracts (low-latency streaming pipeline)

Frozen interfaces for the streaming rework. Subsystems implement these; they do
not redesign them. A change to anything here goes through the integrator.

## Target flow

```
Microphone → capture → VAD + endpointing → Granite ASR        (app/audio, AudioFrontend)
   → Transcript → attention → [history ‖ memory ‖ on-demand vision] → context
   → streaming reasoning → sentence splitter                     (app/orchestration, TurnManager)
   → streaming Phonon TTS → bounded turn-tagged queue → output   (app/tts)
```

One `Turn` exists per request. A newer attention-positive utterance, a barge-in,
or shutdown sets `turn.cancel`; reasoning, synthesis and playback all stop on it.
Ambient vision and memory maintenance never run on this path.

## 1. Request lifecycle, cancellation, correlation, latency — `app/contracts.py`

- `Turn(turn_id, speech_end_time, cancel: threading.Event, marks)`. `turn_id` is
  `SpeechEvent.correlation_id`; it tags every event, log line (`turn=<first 12>`)
  and audio chunk of the request.
- Cancellation is `turn.cancel` (a `threading.Event`) and nothing else. Every
  blocking loop in a worker thread checks it at least once per chunk/token.
- `turn.mark(stage)` records wall-clock `time.time()` for the stages in
  `contracts.STAGES`. `speech_end` is t0 for every latency metric.
- `LatencyRecorder.record(turn, outcome)` stores finished turns;
  `summary()` gives P50/P95/P99 per stage. Benchmarks use `contracts.percentiles`.
- `Transcript` is what the audio frontend emits per utterance.

## 2. Audio frontend — `app/audio/frontend.py` (Agent A)

```python
class AudioFrontend:
    def __init__(self, mic, vad, segmenter, asr, *,
                 on_transcript: Callable[[Transcript], None],
                 on_barge_in: Callable[[float], None] | None = None,
                 barge_in: BargeInConfig, playback_tail_ms: int = 200): ...
    async def run(self) -> None            # long-running; cancel the task to stop
    def set_playback_active(self, active: bool) -> None   # call from the loop thread
    metrics: dict[str, float]
```

- Callbacks are invoked on the event-loop thread and must not block.
- The consumer keeps draining the mic and running VAD while ASR is in flight.
  Finished segments wait in a queue of at most 2 (oldest dropped, counted).
- `set_playback_active(True)`: with barge-in disabled, incoming audio is
  discarded (today's suppression). With barge-in enabled, VAD keeps running at
  `barge_in.vad_threshold` and `on_barge_in(t)` fires once after
  `barge_in.min_speech_ms` of sustained speech.
- `set_playback_active(False)`: resume after `playback_tail_ms`, with VAD and
  segmenter state reset and the mic buffer drained.
- `SpeechSegment` gains `speech_end_time` (capture time of the last speech chunk).
  `minimum_speech_ms` counts speech chunks only. `pre_roll_ms` of audio before
  onset is kept; trailing silence is trimmed before ASR.

## 3. Streaming reasoning — `app/reasoning/` (Agent B)

- `StreamingReasoningModel` in `app/reasoning/model.py` (see docstring):
  `stream(messages, *, max_tokens, temperature, cancel) -> Iterator[str]`,
  `last_stats`, `warm_up()`, `close()`. `generate()` is unchanged and remains
  the full-response fallback.
- `app.reasoning.build_reasoning_model(cfg: ReasoningConfig) -> ReasoningModel`
  is the only constructor the orchestrator calls. It returns the primary Ollama
  model, wrapped with a fallback to `fallback_url`/`fallback_model` when
  `fallback_model` is non-empty. Fallback triggers only when the primary fails
  before its first token; the primary is then skipped for
  `fallback_cooldown_seconds`.
- Timeouts: `connect_timeout_seconds` to connect, `timeout_seconds` between
  stream reads. Requests send `keep_alive` and `think`.

## 4. Streaming TTS and playback — `app/tts/` (Agent C)

- `StreamingTextToSpeech.stream_to(text, sink, *, cancel) -> TTSStreamStats`
  and `StreamingAudioPlayer` in `app/tts/model.py` (see docstrings).
- `PhononTTS` implements `StreamingTextToSpeech`; `load()` runs on its executor.
- `app.tts.player.StreamingPlayer(volume, max_queue_seconds, on_first_audio,
  on_done)` implements `StreamingAudioPlayer`. `on_done(turn_id, outcome)` uses
  the existing `PlaybackOutcome` and fires exactly once per `begin()`.
- The orchestrator calls, per turn: `player.begin(turn_id)`, then for each
  sentence `await tts.stream_to(sentence, partial(player.feed, turn_id=turn_id),
  cancel=turn.cancel)`, then `player.end(turn_id)`.
- Stale audio: zero. A chunk is played only if its `turn_id` is current at the
  moment it reaches the device callback.

## 5. Memory and context — `app/memory/`, `app/context/` (Agent D)

```python
MemoryManager.retrieve_context(query: str, *, limit: int) -> list[str]
MemoryManager.maybe_remember(text: str, source_event_id: str) -> Memory | None
ConversationStore.add_user_message(text, source_event_id)   # idempotent, one commit
```

- All calls are synchronous and are run with `asyncio.to_thread`; they must be
  safe to call concurrently from different threads.
- `retrieve_context` searches stored memories and, when `cross_session` is on,
  messages from earlier sessions, using FTS5/BM25 blended with recency. Each
  returned line carries provenance (kind and date). Target: under 50 ms at 10k rows.
- `maybe_remember` stores an explicit "remember that …" request, deduplicated.
- `ContextBuilder.build(...)` keeps its signature. It keeps the newest turns
  when over budget and never emits the same user utterance twice.

## 6. Orchestration — `app/orchestration/`, `app/main.py` (Agent E)

```python
class TurnManager:
    def __init__(self, *, cfg, bus, attention, context_builder, model,
                 tts=None, player=None, memory=None, fusion=None,
                 frontend=None, recorder: LatencyRecorder, on_demand_vision=None): ...
    async def handle_transcript(self, transcript: Transcript) -> None
    def on_barge_in(self, t: float) -> None
    async def shutdown(self) -> None
```

- `handle_transcript` returns as soon as the turn task is created; the audio
  frontend is never blocked by reasoning.
- Latest wins: a new attention-positive transcript cancels the turn in flight
  (`cancel.set()`, `player.stop()`, task cancelled) before starting its own.
- History, memory retrieval and on-demand vision run concurrently; on-demand
  vision is bounded by `perception.on_demand_timeout_seconds`.
- LLM deltas cross from the worker thread to the loop through a bounded queue;
  sentences reach TTS through a bounded queue. Nothing is unbounded.
- `ResponseEvent` is published once per turn with the full text (console, logs,
  persistence). TTS is not driven through the bus.
- A model that is not a `StreamingReasoningModel` is used through `generate()`.

## 7. Model lifecycle

Every model object has a blocking `load()` (or `warm_up()`) and a `close()`.
Startup installs signal handlers first, loads models off the event loop, drains
the mic buffer, then starts the frontend. Shutdown runs in a `finally`, cancels
the active turn, and closes components in reverse start order.

## 8. Configuration

`configs/default.toml` and `app/config.py` are owned by the integrator. The keys
added for this work are already present; agents read them and report any further
need instead of editing either file.

## File ownership

| Owner | Files |
|---|---|
| Integrator | `app/contracts.py`, `app/config.py`, `configs/`, `pyproject.toml`, `.gitignore`, this document |
| A — audio | `app/audio/`, `tests/test_audio/`, `scripts/benchmark_audio.py` |
| B — reasoning | `app/reasoning/`, `tests/test_reasoning/`, `scripts/benchmark_reasoning.py` |
| C — TTS | `app/tts/`, `tests/test_tts/`, `scripts/benchmark_tts.py` |
| D — memory | `app/memory/`, `app/context/`, `tests/test_memory/`, `tests/test_context/`, `scripts/benchmark_memory.py` |
| E — orchestration | `app/main.py`, `app/orchestration/`, `app/events/`, `app/response/`, `app/attention/`, `app/fusion/`, `app/perception/`, and their tests, `tests/test_integration.py`, `tests/test_orchestration/` |
| F — performance and docs | other `scripts/`, `tests/conftest.py`, `tests/test_scripts/`, `tests/test_config/`, `README.md`, `docs/` except this file |

The protocol signatures in `app/reasoning/model.py` and `app/tts/model.py` are
frozen even for their owners.
