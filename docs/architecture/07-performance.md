# 07 — Performance

How the streaming pipeline's latency is defined, measured and reported. The
interfaces it measures are in [06-engineering-contracts.md](06-engineering-contracts.md).

## Latency stages

Every request is a `Turn` (`app/contracts.py`). `turn.mark(stage)` records a
wall-clock timestamp the first time each stage is reached; all latencies are
milliseconds since `speech_end`. `LatencyRecorder.record(turn, outcome)` stores a
finished turn (outcome `completed`, `cancelled`, `failed` or `no_response`) and
`LatencyRecorder.summary()` gives nearest-rank P50/P95/P99 per stage via
`contracts.percentiles`.

| Stage (`contracts.STAGES`) | Meaning | Marked by |
|---|---|---|
| `speech_end` | Capture time of the last speech chunk; t0 for every metric | `Transcript.speech_end_time` from the audio frontend |
| `segment_emitted` | Endpointing decided the utterance is over | audio frontend (`Transcript.segment_emitted_time`) |
| `asr_done` | Transcript available | audio frontend / orchestrator |
| `attention_done` | Attention decided to reason | orchestrator |
| `context_ready` | History, memory retrieval and on-demand vision assembled | orchestrator |
| `llm_first_token` | First answer token from the streaming model | orchestrator |
| `first_sentence` | First complete sentence handed to TTS | orchestrator |
| `tts_first_chunk` | First synthesized audio chunk | orchestrator (from `stream_to`) |
| `first_audio` | First sample handed to the output device — the user hears the reply | player `on_first_audio` callback |
| `llm_done` | Model finished | orchestrator |
| `playback_done` | Playback drained or was interrupted | player `on_done` callback |

`first_audio` is the headline metric: end of speech to the first audible response.

## Targets

Endpointing is set by configuration: `vad.speech_end_silence_ms` = 192 (six 32 ms
chunks). The memory target comes from the contract. The other targets are proposed
for the streaming pipeline and are to be confirmed with real measurements.

| Metric | Target |
|---|---|
| `speech_end` → `segment_emitted` (endpointing) | < 200 ms |
| `segment_emitted` → `asr_done` (ASR) | < 300 ms |
| `retrieve_context` at 10k rows | < 50 ms |
| `speech_end` → `llm_first_token`, warm | < 1.0 s |
| `speech_end` → `first_audio`, warm | < 1.5 s |
| Cancel (barge-in or newer request) → output silent | < 50 ms (one device block plus scheduling) |
| Stale audio chunks played after cancel | 0 |
| Orchestration overhead, simulated (`first_audio` minus fake model and TTS delays) | < 20 ms |
| Peak process RSS (models on the Mac, LLM remote) | < 2 GB |

## Benchmarks

All are run from the repository root. Scripts that need code not yet in the tree, or
an unreachable host, print `NOT RUN` and exit with code 2.

| Script | What it measures | Notes |
|---|---|---|
| `python -m scripts.benchmark_e2e` | All stages above through the real `TurnManager`, attention, context builder and event bus, with fake streaming model, fake streaming TTS and a fake real-time player (`scripts/e2e_fakes.py`). Also failure rate, max player queue depth, peak RSS and an independent player-observed `first_audio`. | **SIMULATED**: measures orchestration overhead and ordering, not model speed. Fake timing is set by `--ttft-ms`, `--tps`, `--tts-first-chunk-ms`, `--tts-rtf`, `--fail-rate`. `--json PATH` writes results. |
| `python -m scripts.benchmark_e2e --barge-in` | As above, plus barge-in after first audio on every turn: cancel → silence time and stale chunks played. | Simulated. |
| `python -m scripts.benchmark_e2e --real` | Granite ASR, the configured reasoning backend and Phonon TTS. Input speech is synthesized by the TTS; output goes to a null sink (no microphone, no speaker). First turn reported as cold, the rest as warm. | Loads models and calls the Ollama host. |
| `python -m scripts.benchmark_audio` | VAD per-chunk cost, ASR latency and real-time factor, endpointing delay. `--wav` for real speech; `--skip-models` for endpointing only. | |
| `python -m scripts.benchmark_reasoning` | Per target (primary and fallback by default, or `--target url=...,model=...`): cold request, warm time to first token, total latency, tokens/s, failure rate. | Calls Ollama. |
| `python -m scripts.benchmark_tts` | Phonon streaming synthesis with no audio output. | Loads the TTS model. |
| `python -m scripts.benchmark_memory` | `retrieve_context` cold and warm, and message write latency, on synthetic 1k/10k/50k-row databases in a temporary directory. | No models. |
| `python -m scripts.benchmark_attention` | Attention rule and scorer latency. | No models. |
| `python -m scripts.benchmark_vision` | Vision preprocessing and inference with `perception.vision.model` at `perception.vision.ollama_url`. | Calls Ollama. |
| `python -m scripts.stability_test --simulated --turns 300` | Simulated turns through `TurnManager`, a third cancelled by barge-in; fails if live asyncio tasks or threads grow, RSS drifts by 30 MB or more, a turn times out, or a stale chunk plays. | Simulated. |
| `python -m scripts.stability_test --minutes 10` | VAD + segmenter + ASR + event bus on synthetic audio; RSS drift. | Loads VAD and ASR. |

## Results

MacBook Air M4, 16 GB, 2026-11-13, integration branch `feature/jarvis-low-latency-multi-agent`.
Input is speech synthesized by Phonon; no microphone or speaker was used. NOT RUN means not measured.
The primary remote Ollama host was unreachable, so real-model rows use the local fallback `gemma4:12b`.

| Metric | P50 | P95 | P99 | Source |
|---|---|---|---|---|
| Endpointing (`segment_emitted`), 1 s or more of speech | 192 ms | 192 ms | 192 ms | real VAD + segmenter on synthesized speech |
| Endpointing, under 1 s of speech | 480 ms | 480 ms | 480 ms | by configuration (`vad.short_utterance_silence_ms`) |
| ASR (`asr_done` − `segment_emitted`), warm | 40 ms | 46 ms | 48 ms | `benchmark_audio --wav`; cold first call 47–339 ms |
| VAD per 32 ms chunk | 0.08 ms | 0.08 ms | 0.10 ms | `benchmark_audio` |
| `retrieve_context`, 10k rows, warm | 1.12 ms | 1.26 ms | 1.27 ms | `benchmark_memory` |
| `retrieve_context`, 50k rows, warm | 3.35 ms | 6.69 ms | 6.78 ms | `benchmark_memory` |
| Message write | 0.07 ms | NOT RUN | NOT RUN | `benchmark_memory` (P95/P99 not reported to the docs) |
| LLM first token, primary, warm | NOT RUN | NOT RUN | NOT RUN | `benchmark_reasoning` (host unreachable) |
| LLM first token, local fallback, warm, short prompt | 725 ms | 748 ms | 748 ms | `benchmark_reasoning`, n=10; 13.3 tokens/s; cold load 12.5 s |
| LLM first token, local fallback, real system prompt | about 3.1 s | — | — | prompt re-evaluated every turn, see below |
| TTS first chunk, warm | 7.7 ms | 10.2 ms | 10.5 ms | `benchmark_tts --runs 10`; cold 20 ms; real-time factor 0.03 |
| `first_audio`, simulated | 240 ms | 250 ms | 251 ms | `benchmark_e2e --turns 30` (fake model 150 ms to first token, fake TTS 80 ms) |
| `first_audio`, real, warm, local fallback | 5.7 s | 7.8 s | 7.8 s | `benchmark_e2e --real --turns 12`, n=11; excludes endpointing |
| `first_audio`, real, cold (first turn) | 8.2 s | — | — | `benchmark_e2e --real` |
| Cancel → silence, simulated | 6.3 ms | 10.0 ms | 10.0 ms | `benchmark_e2e --turns 30 --barge-in` |

| Check | Result | Source |
|---|---|---|
| Stale chunks played after cancel | 0 in 30 barge-ins | `benchmark_e2e --barge-in` |
| Failure rate (real) | 0 of 12 turns | `benchmark_e2e --real` |
| Max player queue depth | 2.00 s (the configured bound) | `benchmark_e2e` |
| Peak RSS (real) | 525 MB (JARVIS process; Ollama separate) | `benchmark_e2e --real` |
| Task / thread growth, RSS drift | pass over 40 simulated turns; 300 turns NOT RUN | `stability_test --simulated` |

### Where the time goes on the local fallback

With `gemma4:12b` on local Ollama the dominant cost is prompt evaluation, not
generation: about 3 to 4 s for the roughly 300-token prompt on every turn. Ollama
reused its prompt cache only when a request extended the previous request
verbatim (about 1 s in that case). The context builder puts per-turn context
before the conversation history, so the prefix changes every turn and the cache
never applies. Measured on four-turn conversations:

| Prompt layout | Prompt evaluation per turn |
|---|---|
| Current: per-turn context before history | 3.3 to 4.4 s |
| Per-turn context after history | 3.6 to 4.7 s |
| Per-turn context inside the user message, replayed verbatim in history | 4.0 s first turn, then 0.9 to 1.1 s |

Not implemented: the verbatim-replay layout changes what is sent to the model and
has not been validated against the primary model. `think = false` matters: with
thinking left on, time to first token roughly doubled (6.0 s against 3.1 s).

## Local M4 Deployment

Everything below was measured on the target machine itself: MacBook Air M4, 16 GB
unified memory, macOS 27.0.1, Python 3.13.2, Ollama 0.40.15, on 2026-11-13. Nothing
here is simulated. NOT RUN means not measured. Input speech for the automated runs
is synthesized by the local TTS; see "Remaining limitations" for what that leaves
unproven.

### Selected model and runtime

| | |
|---|---|
| Model | Gemma 4 E2B, QAT 4-bit (`Q4_0`, 4.6B parameters) |
| Exact identifier | `gemma4:e2b-it-qat` (Ollama ID `07ea59a47401`) |
| Runtime | Ollama 0.40.15, llama.cpp runner (`llama-server`), 100% GPU (Metal) per `ollama ps` |
| Endpoint | `http://127.0.0.1:11434` |
| Download size | 4.3 GB |
| Context | 4096 tokens (`reasoning.num_ctx`) |

### Why this one

Three candidates were measured, LLM alone, 10 warm runs per prompt size, 4096 context:

| | Ollama `gemma4:e2b-it-qat` | Ollama `gemma4:e2b` | Native MLX `mlx-community/gemma-4-E2B-it-qat-4bit` (mlx-lm 0.32.2) |
|---|---|---|---|
| Download | 4.3 GB | 7.5 GB | 4.36 GB |
| Runner | llama.cpp | MLX (`nvfp4`, 5.2B parameters) | in-process MLX |
| Cold load | 3.6 s | 5.3 s | 2.5 s |
| Model memory | 4.8 GB runner RSS | 7.4 GB footprint | 3.7 GB RSS, 4.1 GB MLX peak |
| System memory free while loaded | 58% | 27% | 54% |
| TTFT, short prompt (~35 tokens), P50 / P95 | 48 / 68 ms | 38 / 124 ms | 135 / 144 ms |
| TTFT, medium prompt (~670 tokens), P50 / P95 | 415 / 588 ms | 1241 / 1300 ms | 719 / 1121 ms |
| TTFT, long prompt (~3120 tokens), P50 / P95 | 1990 / 2223 ms | 2436 / 3712 ms | 3398 / 3767 ms |
| Prompt processing, long prompt, P50 | 1974 ms | 1742 ms | about 910 tokens/s |
| Generation | 55 tokens/s | 58 tokens/s (full pipeline: 44) | 37 tokens/s |
| Streaming | yes | yes | yes |
| Failures | 0 | 0 | 0 |

- On this Mac `ollama pull gemma4:e2b` resolves to a 7.5 GB MLX-tensor build, not
  the 4.6 GB build the registry lists for that tag elsewhere. It gives the fastest
  first audio in the full pipeline (P50 304 ms, P95 379 ms, n=5) but holds 7.4 GB
  and leaves 20% of memory free with ASR and TTS loaded and nothing else open.
  In a six-question check with the real system prompt it also leaked its
  "Thinking Process" text into one answer despite `think = false`; the QAT build
  did not. Answers were otherwise equivalent.
- Native MLX has the smallest footprint but generates more slowly than either
  Ollama build, emits thinking text unless the chat template disables it, and
  would add a second runtime and backend to JARVIS. Not adopted.
- `gemma4:e2b-it-qat` keeps about half the machine free under the full stack,
  stays under the 1.5 s first-audio target, and needs no code beyond the existing
  Ollama client.
- Gemma 4 E4B was not evaluated: E2B meets the latency target and the memory
  headroom is better spent on the rest of the system. `gemma4:12b` and
  `gemma4:e2b` remain installed in Ollama but are not used by the configuration.

8192 context on the selected model: runner RSS 4.9 GB (about the same), medium
prompt TTFT P50 486 ms, long prompt P50 2397 ms. On `gemma4:e2b` the long prompt
went from 2436 ms to 4041 ms. 4096 stays the default; the context builder caps
context at 4000 characters, well inside it.

### Full-stack results on the selected model

`python -m scripts.benchmark_local_stack`, real Granite ASR, Ollama Gemma, Phonon
TTS. Times are milliseconds after end of speech unless stated.

| Scenario | Result |
|---|---|
| A. Gemma alone | TTFT P50 200 ms, P95 216 ms; 54 tokens/s (n=5) |
| B. ASR + Gemma | sequential: ASR 54 ms, TTFT 197 ms. Simultaneous: TTFT 257 ms, concurrent ASR 177 ms (P95 203 ms) |
| C. Gemma + TTS | first audible P50 856 ms sequential, 425 ms when TTS starts while Gemma streams; TTS first chunk 17–20 ms |
| D. ASR + Gemma + TTS | first audible P50 913 ms sequential, 640 ms (P95 1045 ms) simultaneous; concurrent ASR 152 ms |
| E. Full system through `TurnManager` with SQLite memory | first audio P50 535 ms, P95 625 ms (n=10; a 5-turn run gave 443 / 519 ms); ASR done 66 ms; LLM TTFT 288 ms; 54 tokens/s; 10 of 10 completed; max playback queue 2.00 s; 0 stale chunks |
| F. Full system + on-demand local vision | same model analysed a generated test image correctly in 2.5–3.3 s; the vision turn reached first audio at 3.5 s; other turns unaffected (P50 527 ms) |

Scenario E adds the 192 ms endpointing delay in live use, so speech-end to first
audio is about 0.73 s at P50.

Concurrent memory profile, ASR + Gemma + TTS loaded and running (scenario E):

| | |
|---|---|
| JARVIS process | 1.40 GB RSS, 1.66 GB physical footprint |
| Ollama (`llama-server`) | 4.82 GB RSS (weights are file-mapped), 1.67 GB physical footprint |
| Actual peak memory, whole stack | about 6.5 GB |
| System memory free | 59% before, 48% minimum, 57% after |
| Swap used | 2.5 GB before and after, no growth. About 2.7 GB was already in use with no model loaded. |
| CPU | JARVIS mean 20%, max 170% of one core; system mean 44% |
| GPU | model 100% on GPU per `ollama ps`; live Metal utilisation NOT RUN (needs `sudo powermetrics`) |
| Thermal | nominal before and after |
| Model load / warm-up | Gemma cold load 3.6 s; ASR 2.4–2.8 s; TTS 0.15 s |

Sustained generation, 180 s of back-to-back turns: 26 turns, 0 failures, 36
tokens/s in the first minute and 37 in the last, no thermal warning. The Air has
no fan; runs longer than three minutes are NOT RUN.

### Local-only verification

- `runtime.local_only = true` makes `load_config` and `app.main.run` raise
  `ConfigError` for any non-loopback `reasoning.ollama_url`,
  `perception.vision.ollama_url`, or `reasoning.fallback_url` with a fallback model
  set. There is no fallback model by default, so a failed local request raises
  `ReasoningError` naming the URL and the `ollama serve` / `ollama pull` remedy.
- `tests/test_local_only.py` checks: no non-loopback host in the shipped config,
  no Tailscale address or host name in `app/` or `configs/`, no external LLM
  client or API host in `app/`, an explicit failure against a closed local port
  with every attempted connection on loopback, and no `FallbackModel` by default.
- `scripts/test_local_e2e.py` records every socket connection during a full
  run, both in Python and with `lsof` (which also sees native code). Result: the
  only destination was `127.0.0.1:11434`.
- Found and fixed during this work: `onnxruntime` 1.31.1 (Silero VAD) uploads
  telemetry to Microsoft over HTTPS when imported. `app/__init__.py` now sets
  `ORT_DISABLE_TELEMETRY=1`, the documented opt-out; the `lsof` check went from
  failing to clean. It carried no audio or text, but it was an outbound connection.
- Model loading runs with `HF_HUB_OFFLINE=1` in the end-to-end test: all models
  load from the local cache.

### End-to-end test

`python -m scripts.test_local_e2e`: 15 of 15 checks pass, on a WAV fixture fed
through the real VAD, segmenter and ASR.

| Check | Result |
|---|---|
| Transcript | `Jarvis, what is the capital of France?` |
| Attention | reasons, salience 0.95 |
| Memory | seeded fact retrieved into the context |
| Local reasoning, streamed | 7 deltas from `gemma4:e2b-it-qat` at `127.0.0.1:11434` |
| Local TTS, chunk delivery | 6 chunks delivered, turn completed |
| Interruption | barge-in after first audio: outcome `cancelled` |
| Next request | completed; 0 chunks accepted after the cancel, 0 stale blocks played |
| No remote inference | only `127.0.0.1:11434`, Python and `lsof` |

With `--speaker` the same 15 checks pass through the real `StreamingPlayer` and
the Mac's speaker (first audio 644 ms after end of speech).

### Acceptance

| # | Criterion | Status |
|---|---|---|
| 1 | Gemma 4 E2B loads on the M4 | Pass |
| 2 | Reasoning uses localhost | Pass |
| 3 | ASR works locally | Pass |
| 4 | TTS works locally | Pass |
| 5 | Recorded-audio test completes end to end | Pass, on TTS-synthesized WAVs; a human recording is NOT RUN |
| 6 | Live microphone and playback | Playback: pass. Microphone: NOT RUN (needs a person speaking) |
| 7 | Streaming | Pass |
| 8 | Interruption | Pass through the turn manager's barge-in path; a spoken interruption is NOT RUN |
| 9 | Memory retrieval | Pass |
| 10 | No remote inference required | Pass |
| 11 | Within the 16 GB budget | Pass: about 6.5 GB, 48% free at the lowest |
| 12 | Logs confirm runtime and model | Pass: startup logs the policy, model, URL, context and the `/api/ps` residency |
| 13 | Tests document real results | Pass: this section |

### Remaining limitations

- The microphone path has not been exercised. Run
  `python -m scripts.test_local_e2e --mic --speaker` and speak when prompted.
- The WAV fixtures are TTS-synthesized speech (`tests/data/tts_*.wav`), not human
  recordings, and `tests/data/` is gitignored: regenerate them with
  `python -m scripts.test_local_e2e --make-fixture`, or pass `--wav` with a recording.
- Barge-in stays off by default (no echo cancellation); with it off the microphone
  is muted during playback, so a spoken interruption needs headphones and
  `barge_in.enabled = true`.
- `ollama ps` and `/api/ps` report 247 MB for this model because the llama.cpp
  runner file-maps its weights; the 4.8 GB figure above is the runner's RSS.
- Vision is off by default. One analysis takes 2.5–3.3 s against a 3.0 s
  `perception.on_demand_timeout_seconds`, so some on-demand looks will time out
  until that is raised. An ambient analysis already running when a turn starts
  is not interrupted.
- Called directly with the system prompt and no camera, the model still says it
  can see the user. `TurnManager` now adds an explicit "no camera image" note for
  such requests; the system prompt itself still says "You have a camera".
- Memory free figures were taken with few other applications open. A browser
  with many tabs will take some of the 48% headroom.

### Startup

```bash
brew install ollama                 # once
ollama serve                        # or: brew services start ollama
scripts/check_local_environment.sh  # hardware, versions, disk, memory, loopback URLs
scripts/setup_local_models.sh       # pulls reasoning.model only if absent
.venv/bin/python -m scripts.verify_local_gemma
.venv/bin/python -m scripts.test_local_e2e --make-fixture
.venv/bin/python -m scripts.test_local_e2e            # add --speaker and --mic for hardware
.venv/bin/jarvis                    # the assistant itself
```

Benchmarks: `python -m scripts.benchmark_reasoning --num-ctx 4096 --num-ctx 8192`,
`python -m scripts.benchmark_local_stack --scenario all --sustain-seconds 180`, and
for the MLX comparison, in a separate environment with `mlx-lm` installed,
`python scripts/benchmark_mlx_llm.py`.

### Configuration used for the successful test

From `configs/default.toml`:

```toml
[runtime]
local_only = true

[reasoning]
enabled = true
model = "gemma4:e2b-it-qat"
ollama_url = "http://127.0.0.1:11434"
num_ctx = 4096
max_context_messages = 6
max_context_characters = 4000
max_output_tokens = 256
temperature = 0.2
timeout_seconds = 30.0
connect_timeout_seconds = 2.0
keep_alive = "30m"
think = false
fallback_url = "http://127.0.0.1:11434"
fallback_model = ""

[perception.camera]
enabled = false

[perception.vision]
enabled = false
provider = "ollama"
model = "gemma4:e2b-it-qat"
ollama_url = "http://127.0.0.1:11434"
```

`max_concurrent_requests` from the proposed configuration was not added: the turn
manager runs one conversational turn at a time and pauses ambient vision during
it, so there is nothing for the setting to limit.
