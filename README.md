# JARVIS — Audio Perception + Attention Engine

Continuous microphone capture → VAD → speech segmentation → ASR → SpeechEvent → Attention → AttentionDecision.

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
 → console subscriber prints transcript
```

**Phase 1** (audio perception) captures and transcribes speech.
**Phase 2** (attention) decides whether each transcript deserves reasoning resources.

Phase 2 does not generate responses. Phase 2 does not use an LLM. Phase 2 only decides whether an event deserves reasoning. The system remains silent.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run JARVIS (live microphone)
python -m app.main
# Speak into the microphone. Transcripts and attention decisions print to console.
# Ctrl+C to stop.
```

On first run, the ASR model (~1.2 GB) downloads from HuggingFace.

## Performance Targets

| Metric | Target |
|--------|--------|
| VAD latency | <1 ms per 32ms chunk |
| ASR latency | <800 ms from speech-end to transcript |
| End-to-end perception | <1 s speech-end to SpeechEvent |
| Attention (Tier 1 rules) | <0.1 ms |
| Attention (Tier 2 scorer) | <5 ms |
| Total attention | <10 ms |

Run `python -m scripts.benchmark_audio` for audio pipeline measurements.
Run `python -m scripts.benchmark_attention` for attention latency measurements.

## Tests

```bash
# Unit tests (no model download required)
python -m pytest tests/ -v

# All tests including model integration
python -m pytest tests/ -m "integration or not integration" -v
```

Test categories:
- **Unit tests** — no external dependencies, fast
- **Integration tests** (`-m integration`) — require Silero VAD + Qwen3-ASR model downloads
- **Hardware tests** (`-m hardware`) — require real microphone access

## Benchmarks

```bash
pip install psutil
python -m scripts.benchmark_audio
python -m scripts.benchmark_attention
```

## Stability Test

```bash
python -m scripts.stability_test --minutes 10
```

## Configuration

Edit `configs/default.toml`. Key attention settings:

```toml
[attention]
enabled = true
reason_threshold = 0.4
assistant_names = ["jarvis"]
working_memory_size = 20
```

## Architecture

See `docs/architecture/` for the full system design.

## Privacy

- All processing is local. No cloud API calls.
- Raw audio exists only in a bounded in-memory ring buffer.
- No audio or transcripts are persisted to disk.
- Working memory is in-process only, cleared on restart.
- Microphone capture is logged on startup.
