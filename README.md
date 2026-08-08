# JARVIS — Phase 1: Audio Perception Pipeline

Continuous microphone capture → VAD → speech segmentation → ASR → SpeechEvent → EventBus → console.

**Target hardware:** MacBook Air M4, 16 GB RAM, macOS.

## Quick Start

```bash
# Create venv and install
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
  → Silero VAD (512-sample chunks, <1ms)
  → speech segmentation (silence-based boundary detection)
  → Qwen3-ASR 0.6B (MLX, ~340ms per utterance)
  → SpeechEvent on EventBus
  → console subscriber prints transcript
```

Phase 1 only perceives. It does not understand, remember, or respond.

## Tests

```bash
# Unit tests (no model download required)
python -m pytest tests/ -v

# All tests including model integration
python -m pytest tests/ -m "integration or not integration" -v

# Integration test only
python -m pytest tests/test_integration.py -m integration -v
```

## Benchmarks

```bash
pip install psutil
python -m scripts.benchmark_audio
```

## Stability Test

```bash
# 10-minute continuous run (synthetic audio, no microphone needed)
python -m scripts.stability_test --minutes 10
```

## Configuration

Edit `configs/default.toml`:

```toml
[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
ring_buffer_seconds = 30
chunk_samples = 512

[vad]
threshold = 0.5
speech_end_silence_ms = 300
minimum_speech_ms = 250
maximum_segment_seconds = 30

[asr]
model = "Qwen/Qwen3-ASR-0.6B"

[event_bus]
subscriber_queue_size = 256

[logging]
level = "INFO"
```

## Architecture

See `docs/architecture/` for the full system design. Phase 1 implements only the audio perception pipeline — no attention engine, LLM reasoning, memory, TTS, camera, or screen capture.

## Privacy

- All processing is local. No cloud API calls.
- Raw audio exists only in a bounded in-memory ring buffer.
- No audio or transcripts are persisted to disk.
- Microphone capture is logged on startup.
