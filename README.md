# JARVIS

A local-first, low-latency voice assistant for macOS. It listens continuously,
decides which utterances are meant for it, reasons about them with an LLM, and
answers by voice while the answer is still being generated. Speech recognition,
speech synthesis, attention and memory run on the Mac; the LLM runs in Ollama,
by default on a second machine on the local network.

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
- **Reasoning** (`app/reasoning/`): streaming Ollama client with an optional
  local fallback.
- **Speech output** (`app/tts/`): Phonon TTS (`hf://gradium/phonon-31m-en`, voice
  Easton, via the `ptts` runtime) streams audio into one persistent output stream
  fed by a bounded, turn-tagged queue.
- **Vision** (`app/perception/`, optional): camera frames described by a
  vision-capable Ollama model.

One `Turn` exists per request. A newer request, a barge-in or shutdown cancels it,
and reasoning, synthesis and playback stop together. Audio from a cancelled turn is
never played. The interfaces are frozen in
[docs/architecture/06-engineering-contracts.md](docs/architecture/06-engineering-contracts.md);
the full design is in [docs/architecture/](docs/architecture/00-table-of-contents.md).

The orchestration layer (`app/orchestration/turns.py`, `TurnManager`) is being
integrated; until it lands, `app/main.py` still runs the earlier non-streaming path.

## Hardware

- MacBook Air M4, 16 GB, macOS (Apple Silicon is required for MLX ASR).
- Optional remote GPU host running Ollama. The shipped config points
  `reasoning.ollama_url` and `perception.vision.ollama_url` at
  `http://100.97.67.75:11434` (an HP Omen with a GTX 1650 Ti, reached over
  Tailscale) running `gemma4:e2b-it-qat`. Change both URLs to your own host, or to
  `http://127.0.0.1:11434` to run everything on the Mac.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Granite ASR and Phonon TTS weights download from Hugging Face on first use.
On the Ollama host:

```bash
ollama pull gemma4:e2b-it-qat
```

List the Phonon voices with `.venv/bin/ptts --list-voices`.

## Configuration

Everything is in `configs/default.toml` (loaded by `app/config.py`); the system
prompt is `configs/system_prompt.txt`. The keys that matter most:

**Endpointing.** `vad.speech_end_silence_ms` = 192 (six 32 ms chunks of silence end
an utterance), `vad.pre_roll_ms` = 96 (audio kept from before speech onset),
`vad.minimum_speech_ms` = 250.

**Reasoning.** `reasoning.model` = `gemma4:e2b-it-qat` at `reasoning.ollama_url`.
Responses are streamed; `reasoning.connect_timeout_seconds` = 2.0 bounds the
connect and `reasoning.timeout_seconds` = 30.0 bounds each read.
`reasoning.keep_alive` = `30m` keeps the model loaded and `reasoning.think` = false
disables thinking output. `reasoning.max_output_tokens` = 512.

**Local fallback.** If the primary host fails before its first token, the request is
retried on `reasoning.fallback_url` (`http://127.0.0.1:11434`) with
`reasoning.fallback_model` (`gemma4:12b`), and the primary is skipped for
`reasoning.fallback_cooldown_seconds` = 30.0. The fallback only works with Ollama
running on the Mac and the model pulled:

```bash
brew install ollama
ollama serve              # or: brew services start ollama
ollama pull gemma4:12b
```

Set `reasoning.fallback_model` to `""` to disable the fallback.

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
explicit memory. `memory.semantic_enabled` and `memory.embedding_model` are present
but unused: semantic retrieval was evaluated and not implemented.

**Vision.** `perception.enabled`, `perception.camera.enabled` and
`perception.vision.enabled` are all true in the shipped file, so the camera is on.
To turn it off, set `perception.enabled` to false. Frames go to
`perception.vision.model` (`gemma4:e2b-it-qat`) at `perception.vision.ollama_url`;
`perception.camera.analysis_fps` = 0.2 analyses one frame every 5 s.
`perception.on_demand_timeout_seconds` = 3.0 bounds a live look for a vision
question and `perception.pause_during_turn` = true pauses ambient analysis while a
request is being answered.

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
| `python -m scripts.benchmark_e2e --real` | The same with Granite ASR, the configured reasoning backend and Phonon TTS; input speech is synthesized by the TTS, output goes to a null sink. Cold and warm turns reported separately. | Models, Ollama host |
| `python -m scripts.benchmark_audio --skip-models` | Endpointing delay only. Without `--skip-models`: VAD per-chunk cost and ASR latency / real-time factor (`--wav speech.wav` for real speech). | ASR model without `--skip-models` |
| `python -m scripts.benchmark_reasoning` | Cold request, warm time to first token, total latency, tokens/s and failure rate for the primary and fallback targets (`--target url=...,model=...`, `--runs N`). | Ollama |
| `python -m scripts.benchmark_tts` | Phonon streaming synthesis with no audio output (`--runs N`). | TTS model |
| `python -m scripts.benchmark_memory` | `retrieve_context` and message-write latency on synthetic 1k/10k/50k-row databases in a temp directory. | Nothing |
| `python -m scripts.benchmark_attention` | Attention rule and scorer latency. | Nothing |
| `python -m scripts.benchmark_vision` | Vision preprocessing and inference with `perception.vision.model`; prints NOT RUN if the host is unreachable. | Ollama host |
| `python -m scripts.stability_test --simulated --turns 300` | Many simulated turns, a third cancelled; fails on asyncio task, thread or RSS growth. | `app/orchestration/turns.py` |
| `python -m scripts.stability_test --minutes 10` | VAD + segmenter + ASR + event bus on synthetic audio for memory drift. | VAD and ASR models |

Scripts that need code not yet in the tree print `NOT RUN` and exit with code 2.

## Performance

Targets and how each is measured: [docs/architecture/07-performance.md](docs/architecture/07-performance.md).
Measurements are on the MacBook Air M4; NOT RUN means not measured yet. The
endpointing and memory targets come from the config and the engineering contract;
the others are proposed targets for the streaming pipeline.

| Metric | Target | Measured |
|---|---|---|
| Endpointing delay (`speech_end` to `segment_emitted`) | < 200 ms | NOT RUN (192 ms by configuration) |
| ASR (`segment_emitted` to `asr_done`) | < 300 ms | NOT RUN |
| Memory retrieval, 10k rows (warm) | < 50 ms | P50 1.12 ms, P95 1.26 ms, P99 1.27 ms |
| Memory retrieval, 50k rows (warm) | < 50 ms | P50 3.35 ms, P95 6.69 ms, P99 6.78 ms |
| Message write | — | P50 0.07 ms |
| LLM first token (warm, from `speech_end`) | < 1.0 s | NOT RUN |
| First audio (warm, from `speech_end`) | < 1.5 s | NOT RUN |
| First audio (cold, first turn) | — | NOT RUN |
| Barge-in: cancel to silence | < 50 ms | NOT RUN |
| Stale audio chunks played after cancel | 0 | NOT RUN |
| Orchestration overhead (simulated) | < 20 ms | NOT RUN |
| Peak process RSS | < 2 GB | NOT RUN |

## Known limitations

- `TurnManager` is not integrated yet; the running app uses the earlier
  non-streaming path and the end-to-end numbers above are not measured.
- No echo cancellation, so barge-in is off by default (see Configuration).
- The default config depends on a remote Ollama host; without it, replies need the
  local fallback (`ollama serve` plus a pulled model) and are slower on the Mac.
- Memory retrieval is lexical (FTS5); paraphrases with no shared words are missed.
- The shipped config enables the camera. Frames are sent to the vision host over
  the network and are not stored; only text descriptions enter the pipeline.
- Conversation and memories persist in `~/.jarvis/jarvis.db`.
- macOS on Apple Silicon only (MLX).
