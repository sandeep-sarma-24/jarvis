# JARVIS Architecture & Roadmap — Design Spec

**Date:** 2026-08-08 (revised 2026-08-09)
**Status:** Phase 0 APPROVED — Phase 1 may begin
**Full document:** `docs/architecture/` (deliverables A–X split across 5 indexed files)

## Summary

JARVIS is a persistent, local-only, multimodal AI assistant with no wake word. It continuously perceives audio, video, and screen, uses a tiered attention engine to determine whether to spend reasoning resources, maintains a world state model of the current environment, and selectively invokes a local reasoning model. A separate response decision determines whether to speak, store, act, or remain silent. Responses are delivered via TTS in a MCU JARVIS personality.

## Key Architectural Concepts

| Concept | Question It Answers |
|---------|-------------------|
| Attention | "Should JARVIS reason about this?" |
| World State | "What is happening right now?" |
| Memory | "What happened previously?" |
| Context Builder | Assembles state + memories + event for LLM |
| Response Decision | "Should JARVIS actually respond?" |

## Key Decisions

- **Language:** Python (single language for all components)
- **Architecture:** Modular monolith, asyncio, single process
- **Models are replaceable behind interfaces** (none are architectural dependencies)
- **Current reasoning model:** Gemma 4 12B Unified (Q4, ~7.6GB) via Ollama
- **Current ASR:** Qwen3-ASR 0.6B via MLX
- **Current TTS:** Kokoro-82M via kokoro-mlx
- **Current VAD:** Silero VAD v5 (ONNX)
- **Storage:** SQLite (all persistent memory)
- **Event bus:** In-process asyncio queues (correlation-aware)
- **Target hardware:** M4 MacBook Air 16GB (primary), HP Omen GTX 1650 Ti (secondary)
- **Cloud:** None in V1. Reassess when quality is the bottleneck.
- **Performance numbers are targets**, validated empirically, not architectural guarantees.

## Architecture Files

| File | Content |
|------|---------|
| `docs/architecture/00-table-of-contents.md` | Index + quick reference |
| `docs/architecture/01-overview-and-diagrams.md` | Executive overview + Mermaid diagrams (A, B, C, D) |
| `docs/architecture/02-subsystems.md` | Memory, World State, Attention, Perception, Model, Hardware (E, E.1, F, G, H, I) |
| `docs/architecture/03-technology-and-structure.md` | Tech evaluation, repo structure, interfaces (J, K, L) |
| `docs/architecture/04-roadmap-and-operations.md` | Phases, dependencies, testing, metrics, privacy, perf, risks, exit gate (M–S) |
| `docs/architecture/05-vision-and-questions.md` | Open questions (classified), MVP, V1, V2, fully realized (T–X) |

## Open Questions Status

- **BLOCKING for Phase 1:** None
- **RESOLVE before Phase 2:** T5 (Jarvis transcript handling)
- **DEFERRED:** T1 (Phase 4), T2 (Phase 3), T3 (Phase 5+), T4 (Phase 7), T6 (Phase 9), T7 (Phase 4)

## Phase 0 Exit Gate

All 10 criteria met. Phase 1 implementation may begin.
