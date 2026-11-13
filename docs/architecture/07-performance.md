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
