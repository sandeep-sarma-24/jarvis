# JARVIS Architecture & Roadmap — Design Spec

**Date:** 2026-08-08
**Status:** Pending review
**Full document:** `docs/architecture/` (deliverables A–X split across 5 indexed files)

## Summary

JARVIS is a persistent, local-only, multimodal AI assistant with no wake word. It continuously perceives audio, video, and screen, uses a tiered attention engine to filter most input to silence, and selectively invokes a local LLM (Gemma 4 12B) for genuine interactions. Responses are delivered via TTS in a MCU JARVIS personality (dry wit, formal, British).

## Key Decisions

- **Language:** Python (single language for all components)
- **Architecture:** Modular monolith, asyncio, single process
- **Primary LLM:** Gemma 4 12B Unified (Q4, ~7.6GB) via Ollama
- **ASR:** Qwen3-ASR 0.6B via MLX
- **TTS:** Kokoro-82M via kokoro-mlx
- **VAD:** Silero VAD v5 (ONNX)
- **Storage:** SQLite (all persistent memory)
- **Event bus:** In-process asyncio queues
- **Target hardware:** M4 MacBook Air 16GB (primary), HP Omen GTX 1650 Ti (secondary)
- **Cloud:** None in V1. Reassess when quality is the bottleneck.

## Architecture Files

| File | Content |
|------|---------|
| `docs/architecture/00-table-of-contents.md` | Index mapping deliverables A–X to files |
| `docs/architecture/01-overview-and-diagrams.md` | Executive overview + Mermaid diagrams (A, B, C, D) |
| `docs/architecture/02-subsystems.md` | Memory, Attention, Perception, Model, Hardware (E, F, G, H, I) |
| `docs/architecture/03-technology-and-structure.md` | Tech evaluation, repo structure, interfaces (J, K, L) |
| `docs/architecture/04-roadmap-and-operations.md` | Phases, dependencies, testing, metrics, privacy, perf, risks (M–S) |
| `docs/architecture/05-vision-and-questions.md` | Open questions, MVP, V1, V2, fully realized (T–X) |

## Open Questions (require owner decision)

1. TTS voice selection (male British voice may not exist in Kokoro)
2. Ollama vs LiteRT-LM for inference runtime
3. Attention classifier training strategy (when to graduate from weighted scoring)
4. Camera capture approach (always-on vs periodic)
5. "Jarvis" name detection behavior
6. Screen capture permission approach
7. Response length constraints for voice

## Next Step

Review the full architecture document (`docs/architecture/`), resolve open questions, then proceed to implementation planning (Phase 1).
