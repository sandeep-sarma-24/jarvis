# ?A ? Executive Architecture Overview

JARVIS is a persistent, multimodal personal AI assistant that continuously perceives its environment ? audio, video, screen ? and selectively responds through voice. It has no wake word. The system's intelligence lies not in answering questions but in knowing *when* to answer, *when* to remember, and *when* to stay silent.

## Core Design Principle

**Continuous perception, selective reasoning.** The system maintains constant awareness of its environment through cheap, always-running perception pipelines (VAD, frame differencing, screen change detection). Expensive LLM reasoning is invoked only when the attention engine determines something warrants it. The vast majority of perceived input results in silence ? the correct output.

## Architecture in One Paragraph

Audio flows continuously through a VAD ? ASR pipeline. Video flows through adaptive frame sampling ? change detection. Screen capture flows through diff detection ? OCR. All three produce typed **perception events** that feed into a shared **attention engine** ? a tiered system of deterministic rules, a lightweight classifier, and (for genuinely ambiguous cases) the LLM itself. When the attention engine scores an event above threshold, a **context builder** assembles relevant working memory, retrieved episodic/semantic memories, and current environmental state into a compact prompt for the **reasoning model** (Gemma 4 12B, running locally via MLX/Ollama). The model produces a response and a decision (respond/silent/store/act). If "respond," a local TTS engine speaks. If "store," the memory subsystem persists the observation. The system runs as a single Python process (modular monolith) on the primary development machine (M4 MacBook Air, 16GB).

## Key Architectural Decisions

1. **Python monolith** ? single language, single process, asyncio for concurrency. No microservices.
2. **Local-only inference** ? all models run on-device. Cloud fallback deferred to a future reassessment.
3. **Gemma 4 12B Unified** ? encoder-free multimodal model as the primary reasoning engine. Q4 quantized (~7.6GB), fits in 16GB unified memory.
4. **Tiered attention** ? rules first, classifier second, LLM last. Most input never touches the LLM.
5. **SQLite + in-process event bus** ? minimal infrastructure for a single-user, single-machine system.
6. **Voice-first output** ? TTS is the primary response channel. Text UI deferred.
7. **MCU JARVIS personality** ? dry wit, formal but warm, British butler energy.

## Principles Self-Check (Architectural Principle 16)

Tradeoffs explicitly acknowledged:

- **Local-first vs. capability**: Gemma 4 12B at Q4 on 16GB is capable but not frontier-class. Complex multi-step reasoning, long-form code generation, and nuanced instruction-following will be weaker than a cloud model. Accepted: reassess when quality becomes the bottleneck.
- **Privacy vs. memory richness**: All memory is local, which limits cross-device sync and backup. Accepted: single-machine system for now.
- **Modular monolith vs. isolation**: A single process means a crash in the vision pipeline could take down audio. Mitigation: asyncio task isolation + supervisor pattern for critical loops.
- **M4 Air (base) vs. model size**: 16GB constrains context window and concurrent model loading. The 12B model at Q4 leaves ~8GB for OS + applications + KV cache. Context must be kept compact (~2-4K tokens for typical interactions). This makes the context builder's job critical.

---

# ?B ? Complete Component Diagram

```mermaid
flowchart TB
    subgraph Perception["Perception Layer"]
        MIC[Microphone Capture]
        CAM[Camera Capture]
        SCR[Screen Capture]
        VAD[Voice Activity Detection]
        ASR[Speech Recognition]
        FD[Frame Differencing]
        VP[Vision Perception]
        SD[Screen Diff]
        OCR[OCR Engine]
    end

    subgraph Events["Event Bus"]
        EB[Event Bus<br/>In-Process AsyncIO]
    end

    subgraph Attention["Attention Engine"]
        RULES[Deterministic Rules]
        CLF[Lightweight Classifier]
        ATTN_LLM[LLM Judgment<br/>Ambiguous Cases Only]
    end

    subgraph Context["Context Assembly"]
        CB[Context Builder]
        MR[Memory Retrieval]
    end

    subgraph Reasoning["Reasoning Layer"]
        LLM[Gemma 4 12B<br/>Q4 via Ollama/MLX]
        DEC[Decision Layer]
    end

    subgraph Memory["Memory System"]
        WM[Working Memory<br/>In-Process State]
        EM[Episodic Memory<br/>SQLite + Embeddings]
        SM[Semantic Memory<br/>SQLite + Embeddings]
        PM[Procedural Memory<br/>SQLite]
    end

    subgraph Output["Output Layer"]
        TTS[Text-to-Speech<br/>Kokoro/Holler via MLX]
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
    CLF -->|"above threshold"| CB
    CLF -->|"ambiguous"| ATTN_LLM
    ATTN_LLM -->|"reason"| CB
    ATTN_LLM -->|"ignore"| OBS

    CB --> MR
    MR --> WM
    MR --> EM
    MR --> SM
    MR --> PM
    MR --> CB

    CB --> LLM --> DEC

    DEC -->|"respond"| TTS
    DEC -->|"execute tool"| TOOL
    DEC -->|"store"| MEM_W
    DEC -->|"silent"| OBS

    MEM_W --> EM
    MEM_W --> SM
    MEM_W --> PM

    PRIV -.->|"gates"| MIC
    PRIV -.->|"gates"| CAM
    PRIV -.->|"gates"| SCR
    OBS -.->|"traces all"| EB
```

---

# ?C ? Data-Flow Diagram

```mermaid
flowchart LR
    subgraph Input["Raw Input Streams"]
        A1[Audio<br/>16kHz PCM]
        V1[Video<br/>Adaptive FPS]
        S1[Screen<br/>Periodic Capture]
    end

    subgraph Processing["Local Processing"]
        A2[VAD<br/>30ms chunks<br/><1ms CPU]
        A3[ASR<br/>Qwen3-ASR 0.6B<br/>~0.5s latency]
        V2[Frame Diff<br/>Pixel Delta]
        V3[Scene Description<br/>Gemma 4 if needed]
        S2[Screen Diff<br/>Pixel/Hash]
        S3[OCR + App ID]
    end

    subgraph Events["Typed Events"]
        E1[SpeechEvent<br/>transcript, confidence,<br/>is_question, duration]
        E2[VisualEvent<br/>description, objects,<br/>scene_change_magnitude]
        E3[ScreenEvent<br/>app, text_content,<br/>change_type]
    end

    subgraph Attention["Attention Filtering"]
        AT[Attention Engine<br/>? score 0.0-1.0]
    end

    subgraph Context["Context Assembly"]
        CTX[Context Builder<br/>?3000 tokens target]
    end

    subgraph Model["Reasoning"]
        LLM[Gemma 4 12B<br/>~30 tok/s on M4 Air]
    end

    subgraph Output["Actions"]
        OUT[Response / Store / Tool / Silent]
    end

    A1 --> A2 --> A3 --> E1
    V1 --> V2 --> V3 --> E2
    S1 --> S2 --> S3 --> E3

    E1 --> AT
    E2 --> AT
    E3 --> AT

    AT -->|"salience > threshold"| CTX
    AT -->|"salience ? threshold"| OUT

    CTX --> LLM --> OUT
```

---

# ?D ? Event-Flow Diagram

```mermaid
sequenceDiagram
    participant Mic as Microphone
    participant VAD as VAD (Silero)
    participant ASR as ASR (Qwen3-ASR)
    participant Bus as Event Bus
    participant Rules as Rules Engine
    participant Clf as Classifier
    participant CB as Context Builder
    participant Mem as Memory
    participant LLM as Gemma 4 12B
    participant Dec as Decision
    participant TTS as TTS (Kokoro)
    participant Obs as Observability

    Note over Mic,TTS: Typical flow: user speaks, system decides whether to respond

    Mic->>VAD: Continuous 30ms audio chunks
    VAD->>VAD: Speech detected? (< 1ms)
    VAD-->>Obs: No speech ? log, discard

    VAD->>ASR: Speech segment
    ASR->>Bus: SpeechEvent{transcript, confidence, is_question}

    Bus->>Rules: Route event
    Rules->>Rules: Check: explicit address? background? obvious?

    alt Obvious silence (e.g., "I'll call you tomorrow" with no context)
        Rules-->>Obs: ? SILENT (rule: background_conversation)
    else Obvious response (e.g., "Jarvis, what time is it?")
        Rules->>CB: ? RESPOND (rule: explicit_address)
    else Ambiguous
        Rules->>Clf: ? Evaluate with classifier
    end

    Clf->>Clf: Score salience (lightweight model, ~5ms)
    alt Score > 0.7
        Clf->>CB: ? Proceed to reasoning
    else Score 0.3-0.7
        Clf->>LLM: ? Ask LLM to judge (expensive, rare)
    else Score < 0.3
        Clf-->>Obs: ? SILENT (classifier: low salience)
    end

    CB->>Mem: Retrieve relevant memories
    Mem->>CB: Return top-k memories
    CB->>CB: Assemble context (working + episodic + semantic + env)
    CB->>LLM: Prompt with assembled context

    LLM->>Dec: Response + decision metadata
    Dec->>Dec: Parse decision: RESPOND / SILENT / STORE / TOOL

    alt RESPOND
        Dec->>TTS: Generate speech
        TTS->>Mic: Audio output (speaker)
        Dec->>Mem: Store interaction in episodic memory
    else STORE
        Dec->>Mem: Persist observation without responding
    else SILENT
        Dec-->>Obs: Log decision reason
    end

    Dec->>Obs: Log latency, tokens, decision, confidence
```
