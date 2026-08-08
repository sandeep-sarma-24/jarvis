# JARVIS — Multimodal Perception + Attention + Local LLM Reasoning + Voice Output

Audio and visual perception → Attention → Context → Local LLM → ResponseDecision → TTS → Speaker.

**Target hardware:** MacBook Air M4, 16 GB RAM, macOS. Also tested on HP Omen (Ryzen 5 4600H, 8 GB RAM, GTX 1650 Ti).

## Current Pipeline

```
Audio Pipeline:
  Microphone (16kHz mono)
   → PortAudio callback → bounded ring buffer
   → asyncio consumer
   → Silero VAD (512-sample chunks)
   → speech segmentation (silence-based boundary detection)
   → Qwen3-ASR 0.6B (MLX)
   → SpeechEvent on EventBus

Visual Pipeline (opt-in):
  Camera (OpenCV)
   → FrameSampler (bounded queue, latest-frame policy)
   → ChangeDetector (pixel diff)
   → LocalVisionAnalyzer (Gemma 4 12B via Ollama)
     or MockVisualAnalyzer (when vision model disabled)
   → VisualAnalysis → EventDeduplicator
   → VisualEvent on EventBus

Both pipelines feed into:
  → Attention Engine
    → Tier 1: deterministic rules (<0.01ms)
    → Tier 2: weighted scorer (<0.03ms)
  → AttentionDecision {should_reason, salience, reason, tier}
  → if should_reason = false: STOP (no LLM call)
  → if should_reason = true:
      → Context Builder (working memory + conversation + visual context)
      → Reasoning Model (Gemma 4 12B via Ollama)
      → ReasoningResult
      → Response Decision
      → if respond = false: STOP
      → if respond = true:
          → ResponseEvent on EventBus
          → Kokoro TTS (82M, MLX, 24kHz)
          → Audio Playback (sounddevice)
```

**Phase 1** (audio perception) captures and transcribes speech.
**Phase 2** (attention) gates which transcripts deserve reasoning.
**Phase 3** (reasoning) sends attention-positive events through a local LLM and decides whether to respond.
**Phase 4** (voice output) synthesizes and speaks the response via local TTS.
**Phase 5** (working memory) maintains conversation context within a session.
**Phase 6** (visual perception) captures camera frames, detects scene changes, and emits VisualEvents.
**Phase 7** (local vision) sends frames to a local vision model for structured observation.

## Vision Details

```
Vision model:  Gemma 4 12B (via Ollama, same model as reasoning)
Backend:       OllamaVisionBackend (sync httpx + asyncio.to_thread)
Preprocessing: PIL JPEG encode, LANCZOS resize to max 1024px
Output:        Structured JSON → VisualAnalysis (event_type, description, detections)
Privacy:       Frames sent only to localhost Ollama, never persisted
```

Vision is disabled by default. Enable in `configs/default.toml`:
```toml
[perception]
enabled = true

[perception.camera]
enabled = true

[perception.vision]
enabled = true
```

Measured latency: run `python -m scripts.benchmark_vision` with Ollama running.

## TTS Details

```
TTS model:     Kokoro-82M (bf_emma voice, British English)
TTS runtime:   kokoro-mlx (MLX, Apple Silicon native)
Model size:    ~327 MB (bf16 weights)
Sample rate:   24 kHz native output
Interruption:  AudioPlayer.stop() — immediate cancellation
Playback:      sounddevice OutputStream, non-blocking background thread
```

Measured latency values: run `python -m scripts.benchmark_tts` after model download.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install and start Ollama, then pull the model (~7.6 GB)
# gemma4:12b is used for both reasoning and vision
brew install ollama
brew services start ollama
ollama pull gemma4:12b

# Run JARVIS (live microphone, camera off by default)
python -m app.main
# Speak into the microphone. Ctrl+C to stop.
```

On first run, the ASR model (~1.2 GB) and TTS model (~327 MB) download from HuggingFace.

## Performance

Measured on MacBook Air M4, 16 GB RAM, Ollama v0.32.6, gemma4:12b (7.6 GB Q4).

| Metric | Target | Measured |
|--------|--------|----------|
| VAD latency | <1 ms per 32ms chunk | — (see benchmark) |
| ASR latency | <800 ms from speech-end | — (see benchmark) |
| End-to-end perception | <1 s speech-end to event | — (see benchmark) |
| Attention (Tier 1 rules) | <0.1 ms | <0.01 ms |
| Attention (Tier 2 scorer) | <5 ms | <0.03 ms |
| Total attention | <10 ms | <0.05 ms |
| Context assembly | <50 ms | <0.003 ms |
| Model cold start | — | 7,545 ms |
| Model warm inference (mean) | <2 s | 8,144 ms |
| Model warm inference (min) | — | 5,494 ms |
| Model warm inference (max) | — | 8,820 ms |
| Tokens/sec | — | 11.3 tok/s |
| TTS model load | — | 1,126 ms |
| TTS cold synthesis | — | 2,218 ms |
| TTS warm P50 | — | 358 ms |
| TTS warm P95 | — | 518 ms |
| TTS real-time factor | <1.0x | 0.095x |
| TTS model memory | — | ~370 MB |
| Vision preprocessing | <10 ms | 2.8 ms avg |
| Vision inference (mean) | — | 38,808 ms |
| Vision inference (min) | — | 25,774 ms |
| Vision inference (p95) | — | 48,625 ms |
| Vision inference (max) | — | 50,339 ms |
| Vision success rate | 100% | 10/10 |

Gemma 4 12B is a thinking model — inference includes an internal `<think>` phase before producing visible output. The 2s target is not met; typical warm responses take 5–9s. A smaller model or non-thinking variant would be needed to hit the target. Python RSS stays low (~29 MB); Ollama holds the model in its own process.

Kokoro-82M warm synthesis is ~10x faster than real-time on M4. Cold synthesis includes pipeline initialization and voice loading.

Vision inference is 26–50s per frame with gemma4:12b. This is acceptable for ambient perception (the analysis loop uses latest-frame policy, not queued inference), but a smaller model would improve responsiveness. Vision preprocessing (resize + JPEG encode) is negligible at <5ms.

Run `python -m scripts.benchmark_audio` for audio pipeline measurements.
Run `python -m scripts.benchmark_attention` for attention latency measurements.
Run `python -m scripts.benchmark_reasoning` for reasoning pipeline measurements.
Run `python -m scripts.benchmark_vision` for vision pipeline measurements (requires Ollama + gemma4:12b).

## Tests

```bash
# Unit tests (no model download required, no Ollama required)
python -m pytest tests/ -v

# Perception tests only (camera, sampler, vision, lifecycle)
python -m pytest tests/test_perception/ -v

# Local vision tests only (47 tests, fake backend, no Ollama)
python -m pytest tests/test_perception/test_local_vision.py -v

# All tests including model integration
python -m pytest tests/ -m "integration or not integration" -v

# Reasoning model smoke test (requires Ollama + gemma4:12b)
python -m pytest tests/test_reasoning/test_local_model.py -m model -v

# TTS model smoke test (requires Kokoro model download)
python -m pytest tests/test_tts/test_pipeline.py -m model -v
```

Test categories:
- **Unit tests** — no external dependencies, fast (502 tests)
- **Integration tests** (`-m integration`) — require Silero VAD + Qwen3-ASR model downloads
- **Model tests** (`-m model`) — require running Ollama instance or Kokoro TTS model
- **Hardware tests** (`-m hardware`) — require real microphone/speaker access

## Benchmarks

```bash
pip install psutil
python -m scripts.benchmark_audio
python -m scripts.benchmark_attention
python -m scripts.benchmark_reasoning
python -m scripts.benchmark_tts
python -m scripts.benchmark_vision   # requires Ollama + gemma4:12b
```

## Stability Test

```bash
python -m scripts.stability_test --minutes 10
```

## Configuration

Edit `configs/default.toml`. Key settings:

```toml
[attention]
enabled = true
reason_threshold = 0.4
assistant_names = ["jarvis"]
working_memory_size = 20

[reasoning]
enabled = true
model = "gemma4:12b"
max_context_messages = 8
max_context_characters = 6000
max_output_tokens = 256
temperature = 0.2
max_concurrent_requests = 1

[response]
enabled = true

[tts]
enabled = true
model = "mlx-community/Kokoro-82M-bf16"
voice = "bf_emma"
speed = 1.0
sample_rate = 24000
max_text_length = 2000

[playback]
enabled = true
volume = 1.0

[perception]
enabled = false          # camera requires explicit opt-in
max_visual_events = 5

[perception.camera]
enabled = false
device_index = 0
capture_fps = 10.0
analysis_fps = 1.0

[perception.vision]
enabled = false          # enable to use local vision model
provider = "ollama"      # only "ollama" is supported
model = "gemma4:12b"
timeout_seconds = 120.0
max_image_size = 1024
prompt = ""              # empty = use default structured prompt
```

System prompt: `configs/system_prompt.txt` (versioned, editable).

## Architecture

See `docs/architecture/` for the full system design.

### Subsystem boundaries

```
Audio Perception (Phase 1) → never calls the LLM
Visual Perception (Phase 6) → Camera → FrameSampler → ChangeDetector → VisualAnalyzer
Local Vision (Phase 7)      → LocalVisionAnalyzer → OllamaVisionBackend → structured observation
Attention (Phase 2)         → gates LLM access for both speech and visual events
Context Builder             → controls what the LLM sees (speech + conversation + visual context)
ReasoningModel              → replaceable behind protocol
Response Decision           → separate from reasoning
TTS (Phase 4)               → downstream of ResponseDecision, replaceable behind protocol
AudioPlayer                 → stop() for interruption, does not own VAD
```

Reasoning does not block audio capture. LLM inference runs in a thread via `asyncio.to_thread`.
TTS synthesis runs in a thread via `asyncio.to_thread`. Playback runs in a background thread.
Vision inference runs in a thread via `asyncio.to_thread`. Camera capture runs independently.

Phase 3 concurrency policy: at most one local LLM inference is active. If another attention-positive event arrives while inference is active, the event is dropped rather than creating unbounded inference tasks.

Phase 7 vision model: `LocalVisionAnalyzer` sits behind the `VisualAnalyzer` protocol. The `VisionBackend` protocol decouples from Ollama. At most one visual inference runs at a time (single analysis loop + latest-frame policy). Vision uses the same Ollama model as reasoning but through a separate endpoint (`/api/chat` with `images` field).

### Key protocols

```
VisualAnalyzer  → MockVisualAnalyzer (testing) or LocalVisionAnalyzer (production)
VisionBackend   → OllamaVisionBackend (production) or FakeVisionBackend (testing)
ReasoningModel  → OllamaModel (production)
Camera          → OpenCVCamera (production) or FakeCamera (testing)
```

## Privacy

- All processing is local. No cloud API calls.
- Raw audio exists only in a bounded in-memory ring buffer.
- No audio, transcripts, prompts, or responses are persisted to disk.
- TTS audio exists only in memory during playback; no temp files.
- Working memory is in-process only, cleared on restart.
- Model prompts are not logged at INFO level.
- Microphone capture is logged on startup.
- Camera is disabled by default (explicit opt-in via config).
- Raw camera frames are never persisted to disk, logs, or SQLite.
- Frames are sent only to localhost Ollama as base64, discarded after inference.
- Only structured descriptions (text) enter the reasoning pipeline — never raw image data.
- VisualEvents contain text descriptions and metadata, never pixel data.
