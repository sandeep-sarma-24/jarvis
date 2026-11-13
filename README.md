# JARVIS

A local-first, low-latency voice assistant for macOS. It listens continuously,
decides which utterances are meant for it, reasons about them with an LLM, and
answers by voice while the answer is still being generated. Speech recognition,
speech synthesis, attention, memory and the LLM (Gemma 4 E2B in a local Ollama)
all run on the Mac. Remote model hosts are refused at startup.

## Architecture

```
Microphone → capture → VAD + endpointing → Granite ASR        (app/audio, AudioFrontend)
   → Transcript → attention → [history ‖ memory ‖ on-demand vision] → context
   → streaming reasoning → sentence splitter                     (app/orchestration, TurnManager)
   → streaming Phonon TTS → bounded turn-tagged queue → output   (app/tts)
```

- **Audio** (`app/audio/`): PortAudio capture into a bounded ring buffer, Silero
  VAD on 32 ms chunks, silence-based endpointing, then
  `ibm-granite/granite-speech-5.0-470m-turboctc` via mlx-audio.
- **Attention** (`app/attention/`): deterministic rules plus a weighted scorer
  decide whether a transcript deserves reasoning. Most speech stops here.
- **Context** (`app/context/`, `app/memory/`): recent conversation, memories and
  earlier sessions' messages retrieved with SQLite FTS5, and optional visual context.
- **Reasoning** (`app/reasoning/`): streaming client for the local Ollama, with an
  optional second Ollama target (disabled by default).
- **Speech output** (`app/tts/`): Phonon TTS (`hf://gradium/phonon-31m-en`, voice
  Easton, via the `ptts` runtime) streams audio into one persistent output stream
  fed by a bounded, turn-tagged queue.
- **Vision** (`app/perception/`, optional): camera frames described
  by the same local Ollama model that does reasoning.

One `Turn` exists per request. A newer request, a barge-in or shutdown cancels it,
and reasoning, synthesis and playback stop together. Audio from a cancelled turn is
never played. The interfaces are frozen in
[docs/architecture/06-engineering-contracts.md](docs/architecture/06-engineering-contracts.md);
the full design is in [docs/architecture/](docs/architecture/00-table-of-contents.md).

`app/main.py` is thin wiring: `AudioFrontend` feeds `TurnManager`
(`app/orchestration/turns.py`), which drives the streaming model, TTS and player.

## Hardware

- MacBook Air M4, 16 GB, macOS (Apple Silicon is required for MLX ASR).
- Ollama on the same Mac, at `http://127.0.0.1:11434`. No second machine.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Granite ASR and Phonon TTS weights download from Hugging Face on first use.
The LLM runs in a local Ollama:

```bash
brew install ollama
ollama serve              # or: brew services start ollama
ollama pull gemma4:e2b-it-qat
```

If Ollama is not running, JARVIS still starts, logs an ERROR naming the URL and
these commands, and every reply fails with a logged reasoning error until Ollama is
up. It never tries another host.

List the Phonon voices with `.venv/bin/ptts --list-voices`.

## Configuration

Everything is in `configs/default.toml` (loaded by `app/config.py`); the system
prompt is `configs/system_prompt.txt`. The keys that matter most:

**Endpointing.** `vad.speech_end_silence_ms` = 192 (six 32 ms chunks of silence end
an utterance), `vad.pre_roll_ms` = 96 (audio kept from before speech onset),
`vad.minimum_speech_ms` = 250. Utterances with less than
`vad.short_utterance_ms` = 1000 of speech wait `vad.short_utterance_silence_ms` = 480
instead, so the pause after a wake word ("Jarvis, ...") does not split the request.
A pause longer than 192 ms later in a sentence can still split it.

**Local only.** `runtime.local_only` = true: every model URL
(`reasoning.ollama_url`, `perception.vision.ollama_url`, and `reasoning.fallback_url`
when a fallback model is set) must be loopback (`127.0.0.1`, `localhost`, `::1`), or
startup fails with a `ConfigError`. Set it to false only to use a remote host on purpose.

**Reasoning.** `reasoning.model` = `gemma4:e2b-it-qat` at `reasoning.ollama_url`
(`http://127.0.0.1:11434`). `reasoning.num_ctx` = 4096 sets the context window
(the KV cache grows with it). Responses are streamed; `reasoning.connect_timeout_seconds` = 2.0 bounds the
connect and `reasoning.timeout_seconds` = 30.0 bounds each read.
`reasoning.keep_alive` = `30m` keeps the model loaded and `reasoning.think` = false
disables thinking output. `reasoning.max_output_tokens` = 256.

**Fallback.** `reasoning.fallback_model` = `""` disables it (the default: one model
in memory on a 16 GB machine). When set, a request that fails before its first token
is retried on `reasoning.fallback_url` with that model, and the primary is skipped for
`reasoning.fallback_cooldown_seconds` = 300.0.

**Playback.** `playback.max_queue_seconds` = 2.0 bounds synthesized-but-unplayed
audio; `playback.tail_ms` = 200 keeps the microphone suppressed briefly after
playback ends.

**Barge-in.** `barge_in.enabled` = false. With it on, speech during playback
(`barge_in.vad_threshold` = 0.8, sustained for `barge_in.min_speech_ms` = 256)
stops JARVIS mid-sentence. It is off by default because there is no echo
cancellation: through laptop speakers the microphone hears JARVIS's own voice and
would interrupt it. Enable it only with headphones. While it is off the microphone
is suppressed during playback. Speaking while JARVIS is still thinking cancels the
earlier request either way.

**Memory.** `memory.enabled` = true stores conversation and memories in SQLite at
`memory.database_path` (`~/.jarvis/jarvis.db`); an existing database is migrated
in place on first open. Retrieval is FTS5/BM25 blended with recency, up to
`memory.max_retrieved_memories` = 5 lines. `memory.cross_session` = true also
searches messages from earlier sessions. Saying "remember that ..." stores an
explicit memory. Semantic (embedding) retrieval was evaluated and not implemented.

**Vision.** On: `perception.camera.enabled` = true and
`perception.vision.enabled` = true (with `perception.enabled` = true). Set the two
camera/vision keys to false to keep the camera closed. When a request is a vision
request ("what do you see"), typed with "use camera", or carries an uploaded photo,
the JPEG goes straight to the reasoning model as `images` on the user message.
Ambient frames go to
`perception.vision.model` (`gemma4:e2b-it-qat`, the reasoning model, so no second model is
loaded) at `perception.vision.ollama_url`, with the reasoning `num_ctx` and
`keep_alive` so Ollama does not reload the model; a different vision model logs a
warning that two models will be resident.
`perception.camera.analysis_fps` = 0.2 analyses one frame every 5 s.
`perception.on_demand_timeout_seconds` = 3.0 bounds a live look for a vision
question and `perception.pause_during_turn` = true pauses ambient analysis while a
request is being answered.

**Web UI.** `ui.enabled` = true serves a browser UI at http://127.0.0.1:8765 (live camera
preview, chat, voice status). It binds `ui.host` = "127.0.0.1" and `ui.port` = 8765; a
non-loopback host is refused while `runtime.local_only` is true. Protocol:
`docs/architecture/08-ui-protocol.md`.

## Running

```bash
python -m app.main        # or: jarvis
```

This opens the microphone (and the camera if enabled). Ctrl+C stops it.

## Tests

```bash
python -m pytest                     # default: no models, no network, no devices
python -m pytest -m integration      # loads Silero VAD and Granite ASR
python -m pytest -m model            # real Ollama requests and Phonon TTS
python -m pytest -m hardware         # real microphone/speaker
```

Every marker declared in `pyproject.toml` (`integration`, `hardware`, `model`) is
skipped unless selected with `-m`; `tests/conftest.py` enforces this.
In a fresh clone `tests/test_attention/test_evaluation.py` fails because
`tests/data/` is gitignored and `tests/data/attention_cases.json` is not committed.

## Benchmarks

| Command | Measures | Needs |
|---|---|---|
| `python -m scripts.benchmark_e2e` | End of speech to first audio through `TurnManager` with fake models (labelled SIMULATED: orchestration overhead and ordering, not model speed). Per-stage P50/P95/P99, failure rate, queue depth, peak RSS. `--barge-in` adds cancel-to-silence time and stale chunks played (must be 0). `--json PATH` writes results. | `app/orchestration/turns.py` |
| `python -m scripts.benchmark_e2e --real` | The same with Granite ASR, the configured reasoning backend and Phonon TTS; input speech is synthesized by the TTS, output goes to a null sink. Cold and warm turns reported separately. | Models, local Ollama |
| `python -m scripts.benchmark_audio --skip-models` | Endpointing delay only. Without `--skip-models`: VAD per-chunk cost and ASR latency / real-time factor (`--wav speech.wav` for real speech). | ASR model without `--skip-models` |
| `python -m scripts.benchmark_reasoning` | Cold request, then warm time to first token, prompt-processing time, tokens/s and failure rate at three prompt sizes, per context size (`--num-ctx N`, repeatable), with Ollama memory, memory pressure and swap (`--target url=...,model=...`, `--runs N`, `--json PATH`). | Ollama |
| `python -m scripts.benchmark_local_stack` | Real ASR + Gemma + TTS together, scenarios A–F (`--scenario`), sequential and simultaneous, with process and Ollama memory, memory pressure, swap, CPU and thermal state; `--sustain-seconds N` for a sustained run. | Models, local Ollama |
| `python -m scripts.test_local_e2e` | Pass/fail run of the whole local pipeline on a WAV (`--make-fixture` writes the TTS-synthesized ones): transcript, attention, memory, streamed reasoning, TTS delivery, cancellation, and that every connection was loopback. `--speaker` and `--mic` use real hardware. | Models, local Ollama |
| `python -m scripts.verify_local_gemma` | One streamed prompt against the configured loopback Ollama; prints model, latency and residency. | Local Ollama |
| `scripts/check_local_environment.sh`, `scripts/setup_local_models.sh` | Check hardware, versions, disk, memory and loopback URLs; pull `reasoning.model` only if absent. | Ollama for the second |
| `python -m scripts.benchmark_tts` | Phonon streaming synthesis with no audio output (`--runs N`). | TTS model |
| `python -m scripts.benchmark_memory` | `retrieve_context` and message-write latency on synthetic 1k/10k/50k-row databases in a temp directory. | Nothing |
| `python -m scripts.benchmark_attention` | Attention rule and scorer latency. | Nothing |
| `python -m scripts.benchmark_vision` | Vision preprocessing and inference with `perception.vision.model`; prints NOT RUN if the host is unreachable. | Ollama host |
| `python -m scripts.stability_test --simulated --turns 300` | Many simulated turns, a third cancelled; fails on asyncio task, thread or RSS growth. | `app/orchestration/turns.py` |
| `python -m scripts.stability_test --minutes 10` | VAD + segmenter + ASR + event bus on synthetic audio for memory drift. | VAD and ASR models |

Scripts that need code not yet in the tree print `NOT RUN` and exit with code 2.

## Performance

Targets and how each is measured: [docs/architecture/07-performance.md](docs/architecture/07-performance.md).
Measurements are on the MacBook Air M4 (16 GB), 2026-11-13, with synthesized speech
as input; NOT RUN means not measured. The
endpointing and memory targets come from the config and the engineering contract;
the others are proposed targets for the streaming pipeline. The LLM, first-audio and
memory rows are for the local `gemma4:e2b-it-qat` default; the full comparison,
memory profile and startup procedure are in the "Local M4 Deployment" section of that
document.

| Metric | Target | Measured |
|---|---|---|
| Endpointing delay (`speech_end` to `segment_emitted`) | < 200 ms | 192 ms (480 ms for utterances with under 1 s of speech) |
| ASR (`segment_emitted` to `asr_done`) | < 300 ms | P50 40 ms, P95 46 ms, P99 48 ms warm |
| Memory retrieval, 10k rows (warm) | < 50 ms | P50 1.12 ms, P95 1.26 ms, P99 1.27 ms |
| Memory retrieval, 50k rows (warm) | < 50 ms | P50 3.35 ms, P95 6.69 ms, P99 6.78 ms |
| Message write | — | P50 0.07 ms |
| TTS first chunk | < 150 ms | P50 7.7 ms, P95 10.2 ms, P99 10.5 ms warm; 20 ms cold |
| LLM first token, local `gemma4:e2b-it-qat`, model only | < 1.0 s | P50 48 ms short prompt, 415 ms at about 670 prompt tokens, 1990 ms at about 3120; 55 tokens/s |
| First audio, real models through `TurnManager` (warm, excludes endpointing) | < 1.5 s | P50 535 ms, P95 625 ms (n=10) |
| Model cold load | — | Gemma 3.6 s, ASR 2.4–2.8 s, TTS 0.15 s |
| Barge-in: cancel to silence (simulated player) | < 50 ms | P50 6.3 ms, P99 10.0 ms |
| Stale audio chunks played after cancel | 0 | 0 (30 simulated barge-ins) |
| Orchestration overhead before reasoning (simulated) | < 20 ms | P99 0.5 ms |
| Peak memory, ASR + Gemma + TTS running | within 16 GB | JARVIS 1.40 GB RSS, Ollama runner 4.82 GB RSS; system memory free 48% at the lowest |

## Known limitations

- `TurnManager` is not integrated yet; the running app uses the earlier
  non-streaming path and the end-to-end numbers above are not measured.
- No echo cancellation, so barge-in is off by default (see Configuration).
- Replies need the local Ollama running with `gemma4:e2b-it-qat` pulled; without it every
  turn fails with a logged reasoning error.
- Memory retrieval is lexical (FTS5); paraphrases with no shared words are missed.
- With vision enabled, frames go only to the local Ollama and are not stored; only
  text descriptions enter the pipeline.
- Conversation and memories persist in `~/.jarvis/jarvis.db`.
- macOS on Apple Silicon only (MLX).
