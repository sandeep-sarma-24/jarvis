# JARVIS Architecture Document — Table of Contents

## File Index

| File | Deliverables | Content |
|------|-------------|---------|
| `01-overview-and-diagrams.md` | A, B, C, D | Executive overview, component diagram, data-flow diagram, event-flow diagram |
| `02-subsystems.md` | E, E.1, F, G, H, I | Memory architecture, World State, attention architecture, multimodal perception, model architecture, hardware/runtime |
| `03-technology-and-structure.md` | J, K, L | Technology evaluation/recommendations, repository structure, interface definitions |
| `04-roadmap-and-operations.md` | M, N, O, P, Q, R, S, Exit Gate | Phase roadmap, dependencies, testing, evaluation metrics, privacy/security, performance budget, risks, Phase 0 exit gate |
| `05-vision-and-questions.md` | T, U, V, W, X | Open questions (classified), MVP definition, V1/V2/fully-realized definitions, principles self-check |

## Deliverable Cross-Reference

| ID | Deliverable | Location |
|----|------------|----------|
| A | Executive architecture overview | `01-overview-and-diagrams.md` A |
| B | Complete component diagram (Mermaid) | `01-overview-and-diagrams.md` B |
| C | Data-flow diagram (Mermaid) | `01-overview-and-diagrams.md` C |
| D | Event-flow diagram (Mermaid) | `01-overview-and-diagrams.md` D |
| E | Memory architecture | `02-subsystems.md` E |
| E.1 | World State architecture | `02-subsystems.md` E.1 |
| F | Attention architecture | `02-subsystems.md` F |
| G | Multimodal perception architecture | `02-subsystems.md` G |
| H | Model architecture | `02-subsystems.md` H |
| I | Hardware/runtime architecture | `02-subsystems.md` I |
| J | Technology comparison and recommendations | `03-technology-and-structure.md` J |
| K | Repository structure | `03-technology-and-structure.md` K |
| L | Interface definitions | `03-technology-and-structure.md` L |
| M | Phase-by-phase roadmap | `04-roadmap-and-operations.md` M |
| N | Dependencies between phases | `04-roadmap-and-operations.md` N |
| O | Testing strategy | `04-roadmap-and-operations.md` O |
| P | Evaluation metrics | `04-roadmap-and-operations.md` P |
| Q | Privacy/security architecture | `04-roadmap-and-operations.md` Q |
| R | Performance/latency budget | `04-roadmap-and-operations.md` R |
| S | Major technical risks | `04-roadmap-and-operations.md` S |
| T | Open architectural questions (classified) | `05-vision-and-questions.md` T |
| U | Recommended MVP definition | `05-vision-and-questions.md` U |
| V | Definition of "V1 JARVIS" | `05-vision-and-questions.md` V |
| W | Definition of "V2 JARVIS" | `05-vision-and-questions.md` W |
| X | Definition of "fully realized system" | `05-vision-and-questions.md` X |

All deliverables A–X are accounted for, plus World State (E.1) and Phase 0 Exit Gate. No gaps.

## Key Architectural Concepts (Quick Reference)

| Concept | Responsibility | Location |
|---------|---------------|----------|
| **Attention** | "Should JARVIS spend reasoning resources on this?" | `attention/` |
| **World State** | "What is happening right now?" (transient) | `state/` |
| **Memory** | "What happened previously?" (persistent) | `memory/` |
| **Working Memory** | Recent conversation turns (session-scoped) | `memory/working.py` |
| **Context Builder** | Assembles world state + memories + event for LLM | `context/` |
| **Reasoning** | LLM generates understanding/response | `reasoning/` |
| **Response Decision** | "Should JARVIS actually respond?" (post-reasoning) | `reasoning/decision.py` |
