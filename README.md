# JARVIS — Audio Perception + Attention + Local LLM Reasoning + Voice Output

Continuous microphone capture → VAD → speech segmentation → ASR → SpeechEvent → Attention → Context → Local LLM → ResponseDecision → TTS → Speaker.

**Target hardware:** MacBook Air M4, 16 GB RAM, macOS.

## Current Pipeline

```
Microphone (16kHz mono)
 → PortAudio callback → bounded ring buffer
 → asyncio consumer
 → Silero VAD (512-sample chunks)
 → speech segmentation (silence-based boundary detection)
 → Qwen3-ASR 0.6B (MLX)
 → SpeechEvent on EventBus
 → Attention Engine
   → Tier 1: deterministic rules (<0.01ms)
   → Tier 2: weighted scorer (<0.03ms)
 → AttentionDecision {should_reason, salience, reason, tier}
 → if should_reason = false: STOP (no LLM call)
 → if should_reason = true:
     → Context Builder (working memory + system prompt + current event)
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

Phase 4 does not implement camera, tools, persistent memory, or autonomous behavior.

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

# Install and start Ollama, then pull the reasoning model (~7.6 GB)
brew install ollama
brew services start ollama
ollama pull gemma4:12b

# Run JARVIS (live microphone)
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

Gemma 4 12B is a thinking model — inference includes an internal `<think>` phase before producing visible output. The 2s target is not met; typical warm responses take 5–9s. A smaller model or non-thinking variant would be needed to hit the target. Python RSS stays low (~29 MB); Ollama holds the model in its own process.

Kokoro-82M warm synthesis is ~10x faster than real-time on M4. Cold synthesis includes pipeline initialization and voice loading.

Run `python -m scripts.benchmark_audio` for audio pipeline measurements.
Run `python -m scripts.benchmark_attention` for attention latency measurements.
Run `python -m scripts.benchmark_reasoning` for reasoning pipeline measurements.

## Tests

```bash
# Unit tests (no model download required, no Ollama required)
python -m pytest tests/ -v

# All tests including model integration
python -m pytest tests/ -m "integration or not integration" -v

# Reasoning model smoke test (requires Ollama + gemma4:12b)
python -m pytest tests/test_reasoning/test_local_model.py -m model -v

# TTS model smoke test (requires Kokoro model download)
python -m pytest tests/test_tts/test_pipeline.py -m model -v
```

Test categories:
- **Unit tests** — no external dependencies, fast
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
```

System prompt: `configs/system_prompt.txt` (versioned, editable).

## Architecture

See `docs/architecture/` for the full system design.

### Phase 3 + 4 boundaries

```
Perception (Phase 1) → never calls the LLM
Attention (Phase 2)  → gates LLM access
Context Builder      → controls what the LLM sees
ReasoningModel       → replaceable behind protocol
Response Decision    → separate from reasoning
TTS (Phase 4)        → downstream of ResponseDecision, replaceable behind protocol
AudioPlayer          → stop() for interruption, does not own VAD
```

Reasoning does not block audio capture. LLM inference runs in a thread via `asyncio.to_thread`.
TTS synthesis runs in a thread via `asyncio.to_thread`. Playback runs in a background thread.

Phase 3 concurrency policy: at most one local LLM inference is active. If another attention-positive event arrives while inference is active, the event is dropped rather than creating unbounded inference tasks. The `reasoning_dropped` metric increments.

Phase 4 playback policy: at most one response plays at a time. A new response stops current playback and plays the latest response. `AudioPlayer.stop()` can be called externally to interrupt speech (Phase 5+ barge-in connection point).

TTS does not own VAD. VAD does not directly own TTS. This separation allows future barge-in without redesigning either subsystem.

Phase 4 does not speak. Phase 4 does not execute tools. Phase 4 does not persist memory.

## Privacy

- All processing is local. No cloud API calls.
- Raw audio exists only in a bounded in-memory ring buffer.
- No audio, transcripts, prompts, or responses are persisted to disk.
- TTS audio exists only in memory during playback; no temp files.
- Working memory is in-process only, cleared on restart.
- Model prompts are not logged at INFO level.
- Microphone capture is logged on startup.
