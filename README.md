# JARVIS — Phase 1: Audio Perception Pipeline

Continuous microphone capture → VAD → speech segmentation → ASR → SpeechEvent → EventBus → console.

**Target hardware:** MacBook Air M4, 16 GB RAM, macOS.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run JARVIS (live microphone)
python -m app.main
# Speak into the microphone. Transcripts print to console.
# Ctrl+C to stop.
```

On first run, the ASR model (~1.2 GB) downloads from HuggingFace.

## What Phase 1 Does

```
Microphone (16kHz mono)
  → PortAudio callback → bounded ring buffer
  → asyncio consumer
  → Silero VAD (512-sample chunks)
  → speech segmentation (silence-based boundary detection)
  → Qwen3-ASR 0.6B (MLX)
  → SpeechEvent on EventBus
  → console subscriber prints transcript
```

Phase 1 only perceives. It does not understand, remember, or respond.

## Performance Targets

| Metric | Target |
|--------|--------|
| VAD latency | <1 ms per 32ms chunk |
| ASR latency | <800 ms from speech-end to transcript |
| End-to-end | <1 s speech-end to SpeechEvent |

Run `python -m scripts.benchmark_audio` to record actual measurements on your hardware.

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
```

## Stability Test

```bash
# Synthetic pipeline stability test (no microphone needed)
python -m scripts.stability_test --minutes 10
```

This uses simulated audio, not a real microphone. See §16 in the implementation spec for real microphone testing.

## Configuration

Edit `configs/default.toml`. See the file for all available settings.

## Architecture

See `docs/architecture/` for the full system design. Phase 1 implements only the audio perception pipeline.

## Privacy

- All processing is local. No cloud API calls.
- Raw audio exists only in a bounded in-memory ring buffer.
- No audio or transcripts are persisted to disk.
- Microphone capture is logged on startup.
