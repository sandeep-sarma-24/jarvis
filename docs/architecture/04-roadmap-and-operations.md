# §M — Phase-by-Phase Roadmap

> **Current stack.** Phase records below describe the models used when each phase
> was built and keep their measurements as history. Since then: ASR is Granite
> Speech 5 (`ibm-granite/granite-speech-5.0-470m-turboctc`, mlx-audio) instead of
> Qwen3-ASR; TTS is Phonon (`hf://gradium/phonon-31m-en`, voice Easton, `ptts`)
> instead of Kokoro; reasoning and vision use `gemma4:e2b-it-qat` on a remote Ollama
> host, with `gemma4:12b` on local Ollama as the reasoning fallback; memory retrieval
> is SQLite FTS5. The streaming low-latency pipeline is specified in
> [06-engineering-contracts.md](06-engineering-contracts.md) and measured as
> described in [07-performance.md](07-performance.md).

## Phase 0: Architecture (This Document)

**Status:** Complete.
**Deliverable:** This architecture document, reviewed and approved.
**Acceptance:** All Phase 1 architectural dependencies are resolved. Non-blocking architectural decisions are explicitly documented as deferred, with a trigger or phase at which they must be revisited.

---

## Phase 1: Microphone → VAD → ASR

**Status:** Complete.

**Objective:** Continuous audio capture with voice activity detection and speech recognition. The system listens, detects speech, and produces transcripts. No intelligence yet — just working perception.

**Scope boundary (Phase 1 ONLY implements):**
```
Microphone → Audio Capture → VAD → Speech Segmentation → ASR → SpeechEvent → EventBus → Console
```

Phase 1 does NOT implement: attention, LLM reasoning, world state, persistent memory, working memory, camera, screen capture, TTS, or tools.

**Requirements:**
- Continuous microphone capture at 16kHz mono
- VAD classifies 30ms chunks as speech/silence in <1ms
- Speech segments are transcribed to text
- Transcripts are emitted as typed events
- System runs indefinitely without memory leaks or resource exhaustion
- Graceful handling of microphone disconnection/reconnection

**Architecture:**
```
PortAudio / sounddevice callback (OS audio thread, NOT asyncio)
  → thread-safe bounded audio buffer (ring buffer)
  → asyncio consumer task
  → VAD (Silero, ONNX, 30ms chunks)
  → Speech segmentation (start/end detection, 300ms silence = end)
  → ASR (Qwen3-ASR 0.6B via mlx-qwen3-asr)
  → SpeechEvent emission to event bus
  → Console logging (for development visibility)
```

**Audio callback constraints:**
The `sounddevice`/PortAudio callback executes on an OS audio thread, outside the
asyncio event loop. It must:
- Receive PCM frames
- Perform minimal copying
- Write to a bounded, thread-safe buffer
- Never perform model inference
- Never block
- Never call heavyweight asyncio operations directly

The asyncio side owns all processing: VAD, segmentation, ASR, event publication.

**Interfaces:**
- `AudioCapture` — provides continuous audio stream
- `VoiceActivityDetector` — processes chunks, emits speech segments
- `SpeechRecognizer` — transcribes audio to text
- `EventBus` — receives and routes SpeechEvents

**Files to create:**
- `app/__init__.py`
- `app/main.py` — asyncio entrypoint, lifecycle
- `app/config.py` — configuration (audio device, sample rate, thresholds)
- `app/audio/__init__.py`
- `app/audio/capture.py` — microphone capture with ring buffer
- `app/audio/vad.py` — Silero VAD wrapper
- `app/audio/asr.py` — Qwen3-ASR wrapper
- `app/events/__init__.py`
- `app/events/bus.py` — asyncio event bus
- `app/events/types.py` — event dataclasses
- `pyproject.toml` — project dependencies
- `configs/default.toml` — default config
- `tests/test_audio/test_vad.py`
- `tests/test_audio/test_asr.py`

**Dependencies:**
- `sounddevice` — audio capture (PortAudio bindings)
- `numpy` — audio buffer operations
- `onnxruntime` — Silero VAD inference
- `silero-vad` — VAD model + utilities
- `mlx-qwen3-asr` — ASR inference (or `mlx`, `mlx-vlm` if using Gemma 4 for ASR)
- `tomllib` — config parsing (Python 3.11+ standard library, no install needed)

**Implementation steps:**
1. Set up `pyproject.toml` with dependencies, Python 3.11+ requirement
2. Implement `AudioCapture`: continuous callback, ring buffer, chunk yielding
3. Implement `VoiceActivityDetector`: load Silero ONNX, process chunks, detect speech start/end
4. Implement `SpeechRecognizer`: load Qwen3-ASR 0.6B, transcribe complete speech segments
5. Implement `EventBus`: typed async pub/sub with simple subscriber registration
6. Wire together in `main.py`: capture → VAD → ASR → event → log
7. Add graceful shutdown (SIGINT/SIGTERM handling)
8. Test with real microphone input

**Tests:**
- Unit: VAD correctly identifies speech vs silence on test audio files
- Unit: ASR produces reasonable transcripts for known audio
- Unit: Event bus delivers events to subscribers
- Integration: Full pipeline from audio file → transcript event
- Stress: Run for 10 minutes continuous, verify no memory growth

**Acceptance criteria:**
- System starts, listens continuously, logs transcripts in real-time
- VAD processing: <1ms per 30ms chunk (measured)
- ASR latency: <1s from end-of-speech to transcript (measured)
- Memory usage: stable over 10-minute run (no growth beyond initial allocation)
- CPU usage: <5% when idle (no speech), <30% when processing speech
- Clean shutdown on Ctrl+C

**Performance criteria (initial benchmark targets — must be validated empirically on M4 Air):**
- VAD: <1ms per 30ms chunk
- ASR: <800ms from speech-end to transcript (target, not guarantee)
- Memory: <500MB total process (models loaded)
- Ring buffer: configurable (default 30 seconds)

Failure to initially hit the 800ms ASR target triggers profiling and evaluation, not architectural redesign.

**Failure modes:**
- Microphone not found → clear error message, exit
- Model download fails → retry with backoff, exit with instructions
- ONNX runtime not available → fallback to PyTorch Silero (slower but functional)
- Audio buffer overflow → drop oldest frames, log warning

**Rollback strategy:** Phase 1 is the foundation. If it fails, the issue is environmental (hardware, model availability). Rollback = fix environment, not revert code.

---

## Phase 2: Speech → Attention

**Status:** Complete.

**Objective:** Given speech transcripts, decide whether they warrant reasoning — not whether to respond. Implement the tiered attention engine (rules + classifier). No LLM reasoning yet — just the gate that answers "should reason?" not "should respond?".

**Note:** Phase 2 does NOT implement response decision — that separation comes in Phase 3. Attention outputs whether an event warrants reasoning; whether to actually speak is decided later.

**Implementation notes:**
- `AttentionDecision` schema: `{should_reason, salience, reason, tier}`. Simpler than the architecture's 6-field spec — `relevance` and `urgency` deferred to Phase 3+ when world state provides context for them.
- Tier 3 (LLM judgment) not implemented — Phase 2 is LLM-free by design. Tier 3 arrives with Phase 3.
- Default threshold: 0.4 (tuned empirically against 57-case evaluation dataset, 96.5% accuracy).
- Scorer weights are heuristic starting values, not empirically optimized. Stored in `ScorerWeights` dataclass, configurable.
- Working memory is a bounded deque (default 20 entries), in-process only, no persistence.
- Attention evaluation latency: p99 < 0.03ms (target was <10ms).

**Requirements:**
- Deterministic rules handle obvious cases (explicit address, clear background)
- Lightweight scoring handles ambiguous cases
- Attention decisions are logged with reasons
- False positive / false negative counters maintained
- System remains silent by default (silence is the correct baseline)

**Architecture:**
```
SpeechEvent (from Phase 1)
  → Attention Engine
    → Tier 1: Rules (pattern matching, <0.1ms)
    → Tier 2: Scorer (feature extraction + weighted scoring, ~5ms)
  → AttentionDecision {should_reason, salience, reason}
  → Log decision
```

**Interfaces:**
- `AttentionEngine.evaluate(event) → AttentionDecision`
- `RulesEngine.check(event) → RuleResult | None`
- `AttentionClassifier.score(event, context) → float`

**Files to create:**
- `app/attention/__init__.py`
- `app/attention/engine.py`
- `app/attention/rules.py`
- `app/attention/classifier.py`
- `app/memory/__init__.py`
- `app/memory/working.py` (minimal: just recent events list)
- `tests/test_attention/test_rules.py`
- `tests/test_attention/test_engine.py`

**Dependencies (new):**
- None beyond Phase 1 (classifier is rule-based scoring initially, no ML dependency)

**Implementation steps:**
1. Define `AttentionDecision` dataclass and `AttentionEngine` interface
2. Implement Tier 1 rules (explicit address detection, question detection, background detection)
3. Implement Tier 2 scorer (weighted feature combination: is_question × 0.3 + contains_tech_terms × 0.2 + recent_interaction × 0.2 + ...)
4. Implement engine orchestrator (rules first → scorer if unclear → decision)
5. Add working memory (recent events list, last response time)
6. Wire into main loop: speech event → attention → log
7. Tune thresholds empirically with real usage

**Tests:**
- Unit: "Hey Jarvis, what time is it?" → should_reason = true, high salience
- Unit: "I'll call you tomorrow" → should_reason = false, background
- Unit: "Why is this API failing?" (user alone, coding context) → should_reason = true
- Unit: "Pass me the salt" (another person detected) → should_reason = false
- Integration: Pipeline from audio → transcript → attention decision

**Acceptance criteria:**
- System correctly classifies >90% of test utterances (curated set of 50)
- False positive rate <10% on background speech
- Attention evaluation: <10ms per event
- `AttentionDecision` logged with human-readable reasons

**Performance criteria:**
- Tier 1 rules: <0.1ms
- Tier 2 scorer: <5ms
- Total attention pipeline: <10ms

**Failure modes:**
- Ambiguous input with no rules matching → defaults to scorer
- Scorer returns mid-range score → defaults to SILENT (conservative)
- All tiers fail → SILENT (never crash on attention failure)

**Rollback strategy:** Attention engine is isolated. If it produces too many false positives, tighten thresholds via config. If it's too aggressive with silence, loosen. No code rollback needed — it's a tuning problem.

---

## Phase 3: Attention → LLM (Reasoning + Response Decision)

**Status:** Complete.

**Objective:** When attention decides to reason (`should_reason = true`), assemble context, run the reasoning model, and explicitly decide whether to respond. This phase implements BOTH reasoning AND response decision — the decision layer separates "I reasoned about this" from "I should respond to this". Text output to console for now (voice output is Phase 4).

**Implementation notes:**
- Model: Gemma 4 12B Q4 via Ollama (configurable). Behind `ReasoningModel` protocol — swappable.
- Context builder: bounded by `max_context_messages` (8) and `max_context_characters` (6000). Current event always preserved. Prefers `should_reason=True` entries from working memory. Current event excluded from historical candidates to prevent duplication.
- Response decision: deterministic Phase 3 policy — attention-positive + successful reasoning = respond. No second LLM call.
- Concurrency: one turn at a time, latest wins. A newer attention-positive utterance cancels the turn in flight (`TurnManager`).
- LLM inference: runs in thread via `asyncio.to_thread` — does not block audio capture.
- System prompt: versioned in `configs/system_prompt.txt`, not hardcoded.
- No TTS, no tools, no persistent memory, no cloud inference.

**Benchmarks (M4 16GB, Ollama v0.32.6, gemma4:12b 7.6GB Q4):**
- Context assembly: 0.003 ms mean.
- Model cold start: 7,545 ms.
- Model warm inference: 5,494–8,820 ms (mean 8,144 ms). 11.3 tok/s.
- Python RSS during inference: ~29 MB (Ollama holds model in its own process).
- Gemma 4 is a thinking model — internal `<think>` phase consumes tokens before visible output. The 2s inference target is not met; a smaller or non-thinking model would be needed.
- `gemma4:12b-mlx` is not an available Ollama tag. Ollama v0.32.6 already uses Metal on Apple Silicon for GGUF models.

**Requirements:**
- Context builder assembles relevant information into a compact prompt
- Gemma 4 12B generates responses via Ollama
- Decision layer parses model output into actions (respond/silent/store)
- System prompt establishes JARVIS personality (dry wit, formal, British butler)
- Responses are concise (optimized for voice delivery)
- Context stays under 3000 tokens to maintain performance on 16GB

**Architecture:**
```
AttentionDecision {should_reason = true}
  → World State Update (minimal in Phase 3: just conversation state)
  → Context Builder
    → Working memory (recent turns, current activity)
    → World State (current state snapshot)
    → System prompt (personality, guidelines)
    → Current event (transcript + metadata)
  → Reasoning Model via Ollama
  → Response Decision
    → RESPOND: output text (console for now)
    → SILENT: log reason (reasoned but chose not to respond)
    → STORE: persist observation
```

**Interfaces:**
- `ContextBuilder.build(event, attention) → AssembledContext`
- `ReasoningModel.generate(context) → ReasoningResult`
- `DecisionParser.parse(model_output) → Decision`

**Files to create:**
- `app/context/__init__.py`
- `app/context/builder.py`
- `app/reasoning/__init__.py`
- `app/reasoning/model.py` — Ollama client
- `app/reasoning/decision.py` — decision parsing
- `app/reasoning/prompts.py` — system prompt templates
- `tests/test_reasoning/test_model.py`
- `tests/test_context/test_builder.py`
- `scripts/setup_models.py` — pull required Ollama models

**Dependencies (new):**
- `httpx` — async HTTP client for Ollama API
- Ollama installed and running (external dependency)
- `gemma4:12b` model pulled in Ollama

**Implementation steps:**
1. Implement Ollama client (async httpx, OpenAI-compatible endpoint)
2. Design system prompt (JARVIS personality, response guidelines, decision format)
3. Implement context builder (working memory + event → structured prompt)
4. Implement decision parser (extract decision from model output)
5. Wire: attention(should_reason=true) → context → model → decision → console output
6. Add timeout handling (model takes too long → abort, log)
7. Tune system prompt for concise, voice-appropriate responses

**Tests:**
- Unit: Context builder produces well-formed prompts under 3000 tokens
- Unit: Decision parser correctly extracts RESPOND/SILENT/STORE from model output
- Integration: Known question → reasonable answer (manual quality check)
- Performance: Response generation <3s end-to-end

**Acceptance criteria:**
- User speaks a question → system generates a relevant response (printed to console)
- Response latency: <3s from end-of-speech to complete response text
- Context size: <3000 tokens for typical interactions
- JARVIS personality is consistent (formal, concise, helpful)
- System correctly decides SILENT when attention passes a non-question observation

**Performance criteria:**
- Context assembly: <50ms
- Model inference (50-token response): <2s
- Total pipeline (speech end → response text): <3s

**Failure modes:**
- Ollama not running → clear error message, retry with backoff
- Model not pulled → run setup script, provide instructions
- Response timeout (>10s) → abort, log, stay silent
- Malformed model output → treat as SILENT, log for debugging

**Rollback strategy:** If Gemma 4 12B is too slow or produces poor quality responses on the M4 Air, fallback to Gemma 4 E4B (smaller, faster, still multimodal) via config change.

---

## Phase 4: LLM → TTS (Voice Output)

**Objective:** Synthesize JARVIS responses as spoken audio using local TTS. Primary deliverable: JARVIS speaks.

**Primary deliverable:** Response text → TTS (Kokoro at the time; now Phonon) → audio playback. JARVIS has a voice.

**Hard dependencies:** Phase 3 (needs response text to speak).

---

## Phase 5: Working Memory

**Objective:** Maintain conversation context across multiple interactions within a session. The system remembers what was said in the last few minutes.

**Primary deliverable:** Multi-turn conversations work. "What did I just ask you?" produces a correct answer.

**Hard dependencies:** Phase 3 (reasoning must be functional).

---

## Phase 6: Persistent Memory

**Objective:** Implement episodic and semantic memory with SQLite. The system remembers across restarts.

**Primary deliverable:** Memories persist, are retrieved by relevance, and inform future responses.

**Hard dependencies:** Phase 5 (working memory provides the write triggers), embedding model configured.

---

## Phase 6: Visual Perception

**Status:** Complete.

**Objective:** Add camera capture, frame sampling, change detection, and visual event emission. The system sees.

**Implementation notes:**
- `Camera` protocol with `OpenCVCamera` (production) and `FakeCamera` (testing).
- `FrameSampler` enforces bounded queue with latest-frame policy. At most one frame reaches analysis per interval.
- `ChangeDetector` uses cheap pixel-diff to skip unchanged frames.
- `EventDeduplicator` suppresses repeated identical visual events within a time window.
- `PerceptionPipeline` runs two independent asyncio tasks (capture + analysis). A slow analyzer never blocks capture.
- `MockVisualAnalyzer` provides deterministic results for testing and when vision model is disabled.
- Camera is disabled by default (privacy). Explicit opt-in via config.

**Hard dependencies:** Phase 2 (attention engine handles visual events).

---

## Phase 7: Local Vision Model

**Status:** Complete.

**Objective:** Send camera frames to a local vision model for structured image understanding. The system interprets what it sees.

**Implementation notes:**
- `LocalVisionAnalyzer` backed by `VisionBackend` protocol (decoupled from Ollama).
- `OllamaVisionBackend` sends base64 JPEG to Ollama `/api/chat` with `images` field.
- Preprocessing: numpy reshape → BGR→RGB swap if needed → PIL LANCZOS resize → JPEG encode. No raw RGB bytes sent.
- Structured JSON output: event_type (enum), description, confidence [0,1], detections list. Parser validates all fields.
- Tolerant JSON parsing (strips markdown fences, extracts embedded JSON) but does not fabricate fake events on parse failure.
- Separate metrics: `vision_inference_count`, `vision_inference_success`, `vision_inference_failure`, `vision_inference_timeout`, `vision_parse_failure`, `vision_backend_failure`.
- Lifecycle: `LocalVisionAnalyzer.close()` propagates to backend. `main.py` calls close on shutdown.
- Provider validation: only `"ollama"` accepted; unsupported providers rejected with clear error.
- Default prompt: used when `VisionConfig.prompt` is empty; custom prompt forwarded as-is when set.
- Default timeout: 120s (gemma4:12b p95 is ~49s on M4 Air).
- Vision is optional — disabled by default. When disabled, `MockVisualAnalyzer` is used. When Ollama is unavailable, analysis returns None and audio continues.
- No cloud APIs, no raw frame persistence, no frame data in logs or events.

**Benchmark (MacBook Air M4, gemma4:12b):**
- Preprocessing: 1.8ms min, 2.8ms avg, 4.2ms max
- Inference: 25.8s min, 38.8s avg, 50.3s max
- Success rate: 10/10
- Model confirmed vision-capable (`"vision"` in Ollama capabilities)

**Hard dependencies:** Phase 2 (attention engine), Phase 6 (camera + perception pipeline).

---

## Phase 8: Multimodal Context

**Objective:** Combine audio, visual, and screen information into unified context for the reasoning model.

**Primary deliverable:** Gemma 4's native multimodal capability is used — the model receives images alongside text when relevant.

**Hard dependencies:** Phase 7 (camera events), Phase 3 (reasoning).

---

## Phase 9: Screen Understanding

**Objective:** Add screen capture, OCR, and application identification. The system knows what you're working on.

**Primary deliverable:** ScreenEvents with app name, window title, and visible text excerpt.

**Hard dependencies:** Phase 2 (attention engine handles screen events).

---

## Phase 10: Tools

**Objective:** The system can execute actions (file operations, terminal, web search, calendar).

**Primary deliverable:** Tool registry, permission system, at least 3 working tools.

**Hard dependencies:** Phase 3 (reasoning produces tool calls), Phase 12 (permissions designed).

---

## Phase 11: Proactive Intelligence

**Objective:** The system occasionally offers help without being asked, based on observed context and memory.

**Primary deliverable:** Proactive suggestions that are welcome, not annoying. Tight threshold.

**Hard dependencies:** Phase 6 (memory), Phase 8 (multimodal context), Phase 2 (attention can initiate without speech trigger).

---

## Phase 12: Privacy/Security

**Objective:** Full privacy controls, audit logs, permission management, secure tool execution.

**Primary deliverable:** User can see all perception state, control permissions, view/delete memories.

**Hard dependencies:** All perception phases (7, 9) must exist to gate.

---

## Phase 13: Evaluation

**Objective:** Systematic evaluation framework for attention accuracy, response quality, memory retrieval.

**Primary deliverable:** Automated test suite that measures false positive/negative rates, response relevance, latency.

**Hard dependencies:** Phase 6+ (needs sufficient functionality to evaluate meaningfully).

---

## Phase 14: Optimization

**Objective:** Profile and optimize for the 16GB memory constraint. Reduce latency, improve throughput.

**Primary deliverable:** Measured improvements in latency budget, reduced memory pressure.

**Hard dependencies:** Phase 13 (must measure before optimizing).

---

## Phase 15: Full Integration

**Objective:** All subsystems working together as a cohesive system. Polish, edge cases, resilience.

**Primary deliverable:** JARVIS runs continuously, handles all input modalities, manages memory, respects privacy, and feels like a capable, discreet assistant.

**Hard dependencies:** All prior phases.

---

# §N — Dependencies Between Phases

```
Phase 0 (Architecture) ─── gates everything

Phase 1 (Audio) ──────────── no dependencies (first implementation)
Phase 2 (Attention) ───────── depends on Phase 1 (needs events to evaluate)
Phase 3 (Reasoning) ──────── depends on Phase 2 (needs attention decisions)
Phase 4 (TTS) ─────────────── depends on Phase 3 (needs text to speak)
Phase 5 (Working Memory) ──── depends on Phase 3 (needs interaction context)
Phase 6 (Persistent Memory) ─ depends on Phase 5 (extends memory system)

Phase 7 (Camera) ─────────── depends on Phase 2 (attention handles visual events)
Phase 8 (Multimodal) ──────── depends on Phase 7 + Phase 3
Phase 9 (Screen) ─────────── depends on Phase 2 (same pattern as camera)

Phase 10 (Tools) ─────────── depends on Phase 3 + Phase 12
Phase 11 (Proactive) ──────── depends on Phase 6 + Phase 8
Phase 12 (Privacy) ────────── depends on Phase 7 + Phase 9 (needs perception to gate)

Phase 13 (Evaluation) ────── depends on Phase 6+ (meaningful metrics)
Phase 14 (Optimization) ───── depends on Phase 13 (measure first)
Phase 15 (Integration) ────── depends on all
```

**Critical path:** 0 → 1 → 2 → 3 → 4 (delivers speaking assistant in 5 phases)

**Parallelizable after Phase 3:** Phases 7 and 9 can be developed in parallel with Phases 5-6 since they follow the same perception pattern. Phase 12 should be designed alongside Phase 7/9 even if implemented later.

**World State rollout:** World State is introduced minimally in Phase 3 (conversation state only) and expanded in Phases 5, 7, and 9 as more modalities come online.

---

# §O — Testing Strategy

## Levels

1. **Unit tests** — Each module tested in isolation with mocked dependencies. Focus on:
   - Attention rules (deterministic, easy to test)
   - Context builder (predictable input → expected output)
   - Memory operations (store → retrieve → verify)
   - Event bus (publish → subscriber receives)

2. **Integration tests** — Cross-module pipelines with real (but controlled) data:
   - Audio file → VAD → ASR → transcript (known audio, expected text)
   - Transcript → Attention → Context → Model → Decision (mocked model for determinism)
   - Memory store → retrieve → verify relevance ranking

3. **End-to-end tests** — Full pipeline with real models (slow, run on demand):
   - Speak a question → system produces a spoken answer
   - Quality evaluated manually or via LLM-as-judge

4. **Stress tests** — Resource usage over time:
   - 10-minute continuous run: memory stable?
   - 100 events/minute: processing keeps up?

## Tooling

- **pytest** — test runner
- **pytest-asyncio** — async test support
- **Audio test fixtures** — short WAV files with known content (committed to repo)
- **Model mocks** — deterministic fake responses for unit/integration tests

## What NOT to test

- Ollama itself (external dependency, tested by its maintainers)
- Model quality (that's evaluation, Phase 13)
- Hardware-specific behavior (camera capture on CI = impossible)

---

# §P — Evaluation Metrics

## Primary Metrics (User Experience)

| Metric | Definition | Target | Measurement |
|--------|-----------|--------|-------------|
| **False positive response** | JARVIS responded when it should have stayed silent | <5% of responses | User dismisses/ignores response |
| **False negative** | JARVIS stayed silent when it should have responded | <10% of opportunities | User repeats/rephrases, says "Jarvis?" |
| **Response relevance** | Response actually addresses the user's need | >90% | LLM-as-judge on logged interactions |
| **Response latency** | Time from end-of-speech to start-of-audio-response | <3s | Instrumented timestamps |
| **Memory retrieval accuracy** | Retrieved memories are relevant to the query | >80% | Manual review of retrieved context |

## Secondary Metrics (System Health)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Perception latency (speech end → event) | <800ms (initial target) | Instrumented |
| Attention decision latency | <10ms (Tier 1+2) | Instrumented |
| Context assembly latency | <50ms | Instrumented |
| Model inference latency (50 tokens) | <2s | Instrumented |
| TTS latency (time to first audio) | <500ms | Instrumented |
| Memory usage (total process) | <12GB peak | OS monitoring |
| CPU usage (idle) | <5% | OS monitoring |
| CPU usage (active speech) | <40% | OS monitoring |

## Key Definitions

**False positive response:** The system generates and speaks a response when the correct action was silence. Examples:
- User is talking to someone else, system interjects
- User is thinking aloud, system interprets as a question
- System responds to TV/music/podcast audio

**False negative:** The system remains silent when the correct action was to respond. Examples:
- User asks a direct question, system doesn't answer
- User is clearly struggling with something the system could help with
- User addresses the system by context (not name), system misses it

These two metrics are in tension. The design bias is toward silence (tolerate higher false negatives over false positives), because an interrupting assistant is worse than a quiet one.

---

# §Q — Privacy/Security Architecture

## Principles

1. **Local-first:** All processing happens on-device. No cloud calls in V1.
2. **Explicit consent:** Each perception modality (mic, camera, screen) requires explicit opt-in.
3. **Visibility:** User can always see what is currently being perceived and what has been remembered.
4. **Control:** User can pause any modality, delete any memory, enter full privacy mode.
5. **No raw storage by default:** Raw audio and video are processed into text events and discarded. Only the extracted information persists.

## Permission Model

| Modality | Default | Stored Data | Raw Data |
|----------|---------|-------------|----------|
| Microphone | ON (required for core function) | Transcripts only | 30s ring buffer in RAM, never persisted |
| Camera | OFF (opt-in) | Text descriptions only | Frames discarded after processing |
| Screen | OFF (opt-in) | App names + text excerpts | Screenshots discarded after OCR |

## Privacy Mode

`privacy_mode = ON` → All perception stops immediately. No audio capture, no camera, no screen capture. Working memory is cleared. The system is fully inert until privacy mode is toggled off.

Activation: Keyboard shortcut (global hotkey), voice command ("Jarvis, privacy mode"), menu bar toggle (future).

## Data Lifecycle

- **Working memory:** Cleared on restart, on privacy mode activation, or after configured TTL (default: 30 minutes of inactivity)
- **Episodic memory:** Subject to decay. User can view and delete. Automated summarization compresses old entries.
- **Semantic memory:** Persistent until contradicted or explicitly deleted by user.
- **Procedural memory:** Persistent until unused for extended period.
- **Audit log:** Records what was perceived and what decisions were made. Does NOT record raw content. Retained for 30 days by default.

## Security Boundaries

- **Tool execution:** Permission-gated. Dangerous tools always require confirmation.
- **File access:** Sandboxed to user-specified directories.
- **Memory access:** No mechanism for external access. Single-user, single-machine.
- **Model prompts:** Never include system prompt in user-visible output. Log prompts only in debug mode.

---

# §R — Performance/Latency Budget

> **Superseded for the streaming pipeline** by [07-performance.md](07-performance.md),
> which lists the current latency stages, targets and benchmarks. The tables below
> are the original Phase 0 budget for the Qwen3-ASR / Gemma 4 12B / Kokoro stack.

> Performance figures are engineering targets based on available benchmarks and must be validated on the target M4 Air hardware during implementation. Failure to initially meet a target triggers profiling and evaluation; it does not automatically invalidate the architecture.

## End-to-End Pipeline (Speech → Voice Response)

| Stage | Target | Notes |
|-------|---------------|-------|
| VAD (speech detection) | <1ms | Per 30ms chunk, CPU |
| End-of-speech detection | 300ms | Silence threshold before declaring speech done |
| ASR (transcription) | 500ms | Qwen3-ASR 0.6B on MLX |
| Event bus routing | <1ms | In-process async |
| Attention (Tier 1 rules) | <0.1ms | Boolean checks |
| Attention (Tier 2 scorer) | <5ms | Feature extraction + scoring |
| Context assembly | 50ms | Memory retrieval + formatting |
| LLM inference (50 tokens) | 1500-2000ms | Gemma 4 12B Q4 @ ~30 tok/s |
| TTS synthesis | 200-500ms | Kokoro-82M, depends on response length |
| Audio playback start | <10ms | Direct to speaker |
| **Total (speech end → audio start)** | **~2.5-3.5s** | **Target: <3.5s for typical interaction** |

## Memory Budget (16GB Machine)

| Component | Memory | Notes |
|-----------|--------|-------|
| macOS + apps | ~4GB | Baseline system usage |
| Gemma 4 12B (Q4, Ollama) | ~9GB | Loaded, KV cache grows with context |
| Qwen3-ASR 0.6B | ~1.5GB | Loaded when speech active |
| Kokoro TTS | ~500MB | Loaded when response needed |
| Silero VAD | ~50MB | Always loaded, tiny |
| Python process + buffers | ~200MB | App code, ring buffers, working memory |
| SQLite + embeddings | ~100MB | In-memory working set |
| **Total (peak)** | **~15.3GB** | **Tight but feasible** |

ponytail: at peak, this leaves <1GB headroom. ASR and TTS are never active simultaneously (you don't speak while it's speaking), so effective peak is ~14.8GB. If memory pressure becomes an issue, ASR can be loaded on-demand (adds ~2s cold start).

## Processing Budget (Per Second)

| Pipeline | Compute | Frequency |
|----------|---------|-----------|
| Audio capture + VAD | CPU, <1% | Continuous (33 chunks/s) |
| ASR | MLX GPU, ~20% for duration | Per speech segment |
| Screen capture + hash | CPU, <1% | Every 2-5 seconds |
| Camera + diff | CPU, <1% | Every 1-5 seconds (adaptive) |
| LLM inference | GPU, 100% for duration | Only when attention fires |
| TTS synthesis | MLX GPU, ~30% for duration | Only when responding |

---

# §S — Major Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **16GB memory too tight** for concurrent models | High | Degraded latency (swapping) | Load ASR/TTS on-demand; reduce KV cache; use E4B fallback |
| **Gemma 4 12B quality insufficient** for complex reasoning at Q4 | Medium | Poor response quality | Cloud fallback (deferred); E4B for simple tasks, 12B for complex; tune prompts |
| **Attention false positives** annoy user | High (initially) | User disables system | Conservative defaults (silence-biased); feedback loop; easy threshold adjustment |
| **ASR accuracy** insufficient for natural speech | Low (Granite Speech 5; originally Qwen3-ASR) | Misunderstood requests | Confirmation heuristic; ask for repetition; 1.7B upgrade path |
| **TTS voice** doesn't match JARVIS personality | Medium | Breaks immersion | Evaluate multiple voices; Qwen3-TTS voice cloning as upgrade |
| **Ollama stability** under continuous use | Low | Service crashes | Supervisor pattern; auto-restart; health checks |
| **macOS permissions** block mic/camera/screen | Low (one-time) | Can't start | Clear setup instructions; test permissions on launch |
| **Model ecosystem churn** (better models appear monthly) | Certain | Architecture must not be model-locked | All models sit behind abstract interfaces; swapping is a config change |
| **Single-process crash** takes down everything | Medium | Full restart required | asyncio exception isolation; crash recovery from SQLite state |
| **Context window too small** for complex interactions | Medium | Truncated memory, lost context | Aggressive summarization; importance-based pruning; keep prompts lean |

---

# Phase 0 Exit Gate

Phase 0 is complete when:

1. Architecture is documented (deliverables A–X present).
2. Phase 1 dependencies are resolved (no blocking questions remain for Phase 1).
3. World State is distinct from Memory.
4. Attention is distinct from Response Decision.
5. Events have correlation semantics.
6. Models are replaceable.
7. Performance values are measurable targets.
8. Non-blocking questions are deferred explicitly.
9. Phase 1 audio/event contracts are defined.
10. Phase 1 can begin without architectural redesign.

**Status: PHASE 0 APPROVED. Phases 1–7 complete.**
