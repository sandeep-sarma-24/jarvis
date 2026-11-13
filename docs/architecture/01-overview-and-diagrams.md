# A ? Executive Architecture Overview

JARVIS is a persistent, multimodal personal AI assistant that continuously perceives its environment ? audio, video, screen ? and selectively responds through voice. It has no wake word. The system's intelligence lies not in answering questions but in knowing *when* to reason, *when* to respond, *when* to remember, and *when* to stay silent.

## Core Design Principle

**Continuous perception, selective reasoning.** The system maintains constant awareness of its environment through cheap, always-running perception pipelines (VAD, frame differencing, screen change detection). Expensive LLM reasoning is invoked only when the attention engine determines something warrants it. The vast majority of perceived input results in silence ? the correct output.

## Architecture in One Paragraph

Audio flows continuously through a VAD ? ASR pipeline. Video flows through adaptive frame sampling ? change detection. Screen capture flows through diff detection ? OCR. All three produce typed **perception events** (which carry correlation IDs for cross-modal grouping) that feed into a shared **attention engine** ? a tiered system of deterministic rules, a lightweight classifier, and (for genuinely ambiguous cases) the LLM itself. The attention engine answers one question: *should the system spend reasoning resources on this?* When yes, events update the **world state** (the system's model of what is happening *right now*), and the **context builder** assembles world state, retrieved persistent memories, and the current event into a compact prompt for the **reasoning model** (currently `gemma4:e2b-it-qat` via Ollama on a GPU host on the local network, with `gemma4:12b` on local Ollama as fallback). The model produces a response, and a separate **response decision** determines the action: respond via TTS, store to memory, execute a tool, or remain silent. Reasoning does not automatically mean responding. The system runs as a single Python process (modular monolith) on the primary development machine (M4 MacBook Air, 16GB).

## Conceptual Flow

```
Perception Events --> Attention ("should reason?") --> World State Update -->
Memory Retrieval + World State + Event --> Context Builder --> Reasoning -->
Response Decision ("should respond?") --> Action (TTS / Store / Tool / Silent)
```

## Key Architectural Decisions

1. **Python monolith** ? single language, single process, asyncio for concurrency. No microservices.
2. **Local-only inference** ? all models run on-device. Cloud fallback deferred to a future reassessment.
3. **Models are replaceable behind interfaces** ? the current implementation uses `gemma4:e2b-it-qat` via Ollama as the reasoning model (local `gemma4:12b` as fallback), Granite Speech 5 for speech recognition, Phonon for TTS, and Silero for VAD. None of these are architectural dependencies; all sit behind abstract interfaces and are swappable via configuration.
4. **Tiered attention** ? rules first, classifier second, LLM last. Most input never touches the LLM. Attention determines whether to *reason*, not whether to *respond*.
5. **Response decision is separate from attention** ? the reasoning model may determine that an event is worth understanding but not worth responding to. "Should reason?" and "should respond?" are distinct questions answered by different components.
6. **World State is distinct from memory** ? world state represents what is happening *now* (transient, in-process). Memory represents what happened *before* (persistent, queryable). Both inform the context builder but serve different purposes.
7. **SQLite + in-process event bus** ? minimal infrastructure for a single-user, single-machine system.
8. **Voice-first output** ? TTS is the primary response channel. Text UI deferred.
9. **MCU JARVIS personality** ? dry wit, formal but warm, British butler energy.
10. **Performance budgets are measurable targets** ? not architectural guarantees. Figures are based on available benchmarks and must be validated empirically on the target hardware.

## Principles Self-Check (Architectural Principle 16)

Tradeoffs explicitly acknowledged:

- **Local-first vs. capability**: The current reasoning model (Gemma 4 E2B, with 12B as local fallback) is capable but not frontier-class. Complex multi-step reasoning will be weaker than a cloud model. Accepted: reassess when quality becomes the bottleneck.
- **Privacy vs. memory richness**: All memory is local, which limits cross-device sync and backup. Accepted: single-machine system for now.
- **Modular monolith vs. isolation**: A single process means a crash in the vision pipeline could take down audio. Mitigation: asyncio task isolation + supervisor pattern for critical loops.
- **M4 Air (base) vs. model size**: 16GB constrains context window and concurrent model loading. The 12B fallback model at Q4 leaves ~8GB for OS + applications + KV cache when it runs on the Mac; the primary model runs on a separate host. Context must be kept compact (~2-4K tokens for typical interactions). This makes the context builder's job critical.

---

# B ? Complete Component Diagram

```mermaid
flowchart TB
    subgraph Perception["Perception Layer"]
        MIC[Microphone Capture<br/>PortAudio callback ? bounded buffer]
        CAM[Camera Capture]
        SCR[Screen Capture]
        VAD[Voice Activity Detection<br/>asyncio consumer]
        ASR[Speech Recognition]
        FD[Frame Differencing]
        VP[Vision Perception]
        SD[Screen Diff]
        OCR[OCR Engine]
    end

    subgraph Events["Event Bus"]
        EB[Event Bus<br/>Bounded AsyncIO Queues<br/>Correlation-aware]
    end

    subgraph Attention["Attention Engine"]
        direction TB
        RULES[Deterministic Rules]
        CLF[Lightweight Classifier]
        ATTN_LLM[LLM Judgment<br/>Ambiguous Cases Only]
    end

    subgraph State["World State"]
        WS[World State<br/>Current environment,<br/>task, activity]
    end

    subgraph Context["Context Assembly"]
        CB[Context Builder]
        MR[Memory Retrieval]
    end

    subgraph Reasoning["Reasoning Layer"]
        LLM[Reasoning Model<br/>Currently: gemma4:e2b-it-qat via Ollama]
    end

    subgraph ResponseDec["Response Decision"]
        RD[Response Decision<br/>respond / store / tool / silent]
    end

    subgraph Memory["Memory System"]
        EM[Episodic Memory<br/>SQLite + Embeddings]
        SM[Semantic Memory<br/>SQLite + Embeddings]
        PM[Procedural Memory<br/>SQLite]
    end

    subgraph Output["Output Layer"]
        TTS[Text-to-Speech]
        TOOL[Tool Executor]
        MEM_W[Memory Writer]
    end

    subgraph Privacy["Privacy & Observability"]
        PRIV[Privacy Controller]
        OBS[Observability<br/>Structured Logs + Metrics]
    end

    MIC --> VAD --> ASR --> EB
    CAM --> FD --> VP --> EB
    SCR --> SD --> OCR --> EB

    EB --> RULES
    RULES -->|"obvious silence"| OBS
    RULES -->|"unclear"| CLF
    CLF -->|"below threshold"| OBS
    CLF -->|"above threshold"| WS
    CLF -->|"ambiguous"| ATTN_LLM
    ATTN_LLM -->|"reason"| WS
    ATTN_LLM -->|"ignore"| OBS

    WS --> CB
    CB --> MR
    MR --> EM
    MR --> SM
    MR --> PM
    MR --> CB

    CB --> LLM --> RD

    RD -->|"respond"| TTS
    RD -->|"execute tool"| TOOL
    RD -->|"store"| MEM_W
    RD -->|"silent"| OBS

    MEM_W --> EM
    MEM_W --> SM
    MEM_W --> PM

    PRIV -.->|"gates"| MIC
    PRIV -.->|"gates"| CAM
    PRIV -.->|"gates"| SCR
    OBS -.->|"traces all"| EB
```

---

# C ? Data-Flow Diagram

```mermaid
flowchart LR
    subgraph Input["Raw Input Streams"]
        A1[Audio<br/>16kHz PCM<br/>callback ? buffer]
        V1[Video<br/>Adaptive FPS]
        S1[Screen<br/>Periodic Capture]
    end

    subgraph Processing["Local Processing"]
        A2[VAD<br/>asyncio consumer<br/>30ms chunks]
        A3[ASR]
        V2[Frame Diff<br/>Pixel Delta]
        V3[Scene Description<br/>LLM if needed]
        S2[Screen Diff<br/>Pixel/Hash]
        S3[OCR + App ID]
    end

    subgraph Events["Typed Events<br/>with correlation_id"]
        E1[SpeechEvent]
        E2[VisualEvent]
        E3[ScreenEvent]
    end

    subgraph Attention["Attention<br/>should reason?"]
        AT[Tiered Attention Engine]
    end

    subgraph StateCtx["World State + Context"]
        WS[World State Update]
        CTX[Context Builder<br/>State + Memories + Event]
    end

    subgraph Model["Reasoning"]
        LLM[Reasoning Model]
    end

    subgraph Decision["Response Decision<br/>should respond?"]
        RD[Response Decision]
    end

    subgraph Output["Actions"]
        OUT[TTS / Store / Tool / Silent]
    end

    A1 --> A2 --> A3 --> E1
    V1 --> V2 --> V3 --> E2
    S1 --> S2 --> S3 --> E3

    E1 --> AT
    E2 --> AT
    E3 --> AT

    AT -->|"should_reason = true"| WS
    AT -->|"should_reason = false"| OUT

    WS --> CTX --> LLM --> RD --> OUT
```

---

# D ? Event-Flow Diagram

```mermaid
sequenceDiagram
    participant Mic as Microphone
    participant VAD as VAD
    participant ASR as ASR
    participant Bus as Event Bus
    participant Rules as Rules Engine
    participant Clf as Classifier
    participant WS as World State
    participant CB as Context Builder
    participant Mem as Memory
    participant LLM as Reasoning Model
    participant RD as Response Decision
    participant TTS as TTS
    participant MW as Memory Writer
    participant Obs as Observability

    Note over Mic,TTS: Typical flow: user speaks, system decides whether to reason, then whether to respond

    Mic->>Mic: PortAudio callback ? bounded buffer
    Note over Mic: OS thread, no asyncio
    Mic->>VAD: Asyncio consumer reads 30ms chunks
    VAD->>VAD: Speech detected?
    VAD-->>Obs: No speech -> discard

    VAD->>ASR: Speech segment
    ASR->>Bus: SpeechEvent{transcript, confidence, correlation_id}

    Bus->>Rules: Route event
    Rules->>Rules: Check: explicit address? background? obvious?

    alt Obvious silence (e.g., "I'll call you tomorrow")
        Rules-->>Obs: SILENT (rule: background_conversation)
    else Obvious reasoning needed (e.g., "Jarvis, what time is it?")
        Rules->>WS: should_reason=true (rule: explicit_address)
    else Ambiguous
        Rules->>Clf: Evaluate with classifier
    end

    Clf->>Clf: Score salience (~5ms)
    alt Score > 0.7
        Clf->>WS: should_reason=true
    else Score 0.3-0.7
        Clf->>LLM: Ask LLM to judge (expensive, rare)
    else Score < 0.3
        Clf-->>Obs: SILENT (classifier: low salience)
    end

    WS->>WS: Update current state (task, activity, entities)
    WS->>CB: Provide current world state
    CB->>Mem: Retrieve relevant memories
    Mem->>CB: Return top-k memories
    CB->>CB: Assemble context (world state + memories + event)
    CB->>LLM: Prompt with assembled context

    LLM->>RD: Reasoning result + response metadata
    RD->>RD: Decide: RESPOND / SILENT / STORE / TOOL

    alt RESPOND
        RD->>TTS: Generate speech
        RD->>MW: Store interaction
    else STORE (relevant but no response needed)
        RD->>MW: Persist observation without responding
    else TOOL
        RD->>RD: Execute tool, then decide on response
    else SILENT (reasoned but nothing to say)
        RD-->>Obs: Log reasoning result + silence reason
    end

    MW->>MW: Importance scoring
    MW->>Mem: Write if importance >= threshold

    RD->>Obs: Log latency, tokens, decision, confidence
```
