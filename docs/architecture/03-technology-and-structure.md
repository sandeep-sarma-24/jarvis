# §J — Technology Comparison and Recommendations

## Voice Activity Detection (VAD)

| Technology | F1 Score | Latency | Memory | License | Apple Silicon | Streaming | Recommendation |
|-----------|----------|---------|--------|---------|--------------|-----------|----------------|
| **Silero VAD v5** | 95.95% | <1ms/chunk (ONNX) | ~50MB | MIT | Yes (CPU) | Yes | **Recommended for V1** |
| FireRedVAD | 97.57% | ~10ms/frame | ~100MB | Apache 2.0 | Via ONNX/NCNN | Yes | Better accuracy, evaluate for V2 |
| WebRTC VAD | 52.30% | <0.1ms | <1MB | BSD | Yes | Yes | Too inaccurate |

**Current implementation: Silero VAD v5.** Most mature, best documented, <1ms on CPU, proven Python integration. FireRedVAD is more accurate (97.57% vs 95.95%) but newer (March 2026) and less ecosystem integration. Upgrade path: swap Silero → FireRedVAD behind the `VoiceActivityDetector` interface once it matures.

Sources: [Silero VAD GitHub](https://github.com/snakers4/silero-vad), [FireRedVAD HuggingFace](https://huggingface.co/FireRedTeam/FireRedVAD)

## Speech Recognition (ASR)

| Technology | WER (LibriSpeech clean) | Latency (M4) | Memory | Languages | License | MLX Native |
|-----------|------------------------|--------------|--------|-----------|---------|-----------|
| Qwen3-ASR 0.6B (original choice) | 2.29% (fp16) | ~460ms (short clip) | 1.2GB | 30+ | Apache 2.0 | Yes (mlx-qwen3-asr) |
| Qwen3-ASR 1.7B | 1.99% | ~4s | 3.4GB | 30+ | Apache 2.0 | Yes |
| Whisper Large v3 | 2.11% | ~2-3s | 3GB | 100+ | MIT | Via whisper.cpp |
| Parakeet TDT | ~2.1% | <500ms | 634MB (CoreML) | 25 | Apache 2.0 | CoreML (Neural Engine) |

**Current implementation: Granite Speech 5 (`ibm-granite/granite-speech-5.0-470m-turboctc`) via mlx-audio** (`app/audio/asr.py`, config `asr.model`). It replaced Qwen3-ASR 0.6B, the original V1 choice evaluated in the table above. Its latency on the M4 Air has not been measured yet; see [07-performance.md](07-performance.md).

Sources: [mlx-qwen3-asr benchmarks](https://github.com/moona3k/mlx-qwen3-asr/blob/main/docs/BENCHMARKS.md), [Qwen3-ASR HuggingFace](https://huggingface.co/collections/Qwen/qwen3-asr)

## Text-to-Speech (TTS)

| Technology | Quality | Latency | Memory | Voices | License | MLX Native | Voice Character |
|-----------|---------|---------|--------|--------|---------|-----------|----------------|
| Kokoro-82M (original choice) | 9.2/10 | 3-5x RT (CPU), ~45ms (CoreML) | ~500MB | 54 | Apache 2.0 | Yes (kokoro-mlx) | Fixed presets (BF_Emma = British female) |
| Qwen3-TTS 0.6B | High | ~120ms TTFA | ~2.3GB (bf16) | Voice cloning | Apache 2.0 | Yes (Holler) | Customizable |
| Holler (Qwen3-TTS fine-tune) | High | 120ms TTFA, 2.5x RTF | 1.7GB (6-bit) | 6 American | Apache 2.0 | Yes | American English, streaming |

**Current implementation: Phonon TTS (`hf://gradium/phonon-31m-en`, voice Easton, q8_0) via the `ptts` runtime** (`app/tts/local_tts.py`, config `[tts]`). It streams audio into a bounded, turn-tagged player (`app/tts/player.py`, `StreamingPlayer`). It replaced Kokoro-82M via kokoro-mlx, the original V1 choice evaluated in the table above. Its latency on the M4 Air has not been measured yet; see [07-performance.md](07-performance.md).

Sources: [kokoro-mlx PyPI](https://pypi.org/project/kokoro-mlx/), [Holler GitHub](https://github.com/sentiuminc/holler)

## Primary LLM (Reasoning)

| Technology | Multimodal | Size (Q4) | Context | tok/s (M4 Air est.) | License |
|-----------|-----------|-----------|---------|---------------------|---------|
| Gemma 4 12B Unified (original choice, now fallback) | Text+Image+Audio | 7.6GB | 256K | ~25-35 | Apache 2.0 |
| Llama 3.1 8B | Text only | 4.5GB | 128K | ~45-55 | Llama |
| Qwen3 14B | Text only | 8.5GB | 128K | ~20-25 | Apache 2.0 |
| Phi-4 14B | Text+Image | 8.5GB | 16K | ~20-25 | MIT |

**Current implementation: `gemma4:e2b-it-qat` on a remote Ollama host** (config `reasoning.model`, `reasoning.ollama_url`), streamed, with an optional local fallback to `gemma4:12b` on `http://127.0.0.1:11434` when the primary fails before its first token (`reasoning.fallback_model`, `reasoning.fallback_url`). The same e2b model serves vision. Gemma 4 12B, evaluated above, was the original choice; it was too slow for conversational latency on the M4 Air (warm responses of 5–9 s with the thinking phase) and is now only the fallback. Context management remains critical — keep prompts under 4K tokens.

Sources: [Google AI Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4), [AI on Mac benchmarks](https://ai-on-mac.com/articles/gemma-4-12b-mac-en/)

## Inference Runtime

| Runtime | tok/s (12B, M4 Air est.) | Multi-model | API | Maturity | Cross-platform |
|---------|--------------------------|-------------|-----|----------|---------------|
| **Ollama** | ~25-30 | Yes | OpenAI-compat | High | Mac/Linux/Win |
| llama.cpp (raw) | ~28-33 | Manual | OpenAI-compat | High | Mac/Linux/Win |
| MLX (mlx-lm) | ~30-38 | Limited | Python | Medium | Mac only |
| LiteRT-LM | ~35-45 (unconfirmed on Air) | Yes | OpenAI-compat | Low (v0.13) | Mac/Linux/Win |

**Current implementation: Ollama for LLM serving**, on a remote GPU host with a local Ollama as fallback. No MLX reasoning backend exists. Best balance of throughput, multi-model support, API stability, and developer experience. The ~10% throughput sacrifice vs raw MLX is worth the operational simplicity. LiteRT-LM is the upgrade candidate once confirmed on base M4.

Sources: [Contra Collective comparison](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026), [Ollama docs](https://ollama.ai)

## Embedding Model

| Model | Dimensions | Size | Latency | Quality (MTEB) |
|-------|-----------|------|---------|----------------|
| nomic-embed-text (evaluated, not used) | 768 | ~500MB | <10ms | High |
| all-MiniLM-L6-v2 | 384 | ~80MB | <5ms | Medium |
| mxbai-embed-large | 1024 | ~1.2GB | ~15ms | Very High |

**Not implemented.** Memory retrieval is lexical: SQLite FTS5/BM25 blended with recency (`app/memory/`). Semantic retrieval was evaluated and not built. This table is kept as the evaluation for a future semantic layer.

## Event Bus / IPC

| Technology | Latency | Persistence | Complexity | Operational Familiarity |
|-----------|---------|-------------|------------|------------------------|
| **asyncio queues (in-process)** | <0.01ms | None (in-memory) | Zero | N/A (stdlib) |
| NATS | ~1ms (local) | Optional (JetStream) | Medium (separate process) | High (author's day job) |
| Redis Pub/Sub | ~0.5ms | No (fire-and-forget) | Low-Medium (separate process) | High (author's day job) |
| ZeroMQ | ~0.1ms | No | Low | Low |

**Current implementation: bounded asyncio queues for V1.** For a single-process Python monolith, in-process async queues are the correct abstraction. Zero infrastructure, zero latency, zero failure modes. The event bus is a typed bounded async queue with subscriber registration. Each subscriber gets a bounded queue (default 256 events); if a subscriber falls behind, the oldest event is dropped and a warning logged. Audio capture must never block because an event consumer is slow.

**When to reconsider NATS:** If the system splits into multiple processes (e.g., a separate perception daemon), NATS becomes the obvious upgrade. The interface (`publish(event)` / `subscribe(pattern, handler)`) is the same shape either way — only the transport changes.

**Redis evaluation:** Redis adds value when you need cross-process shared state with TTLs (exactly what working memory would be in a multi-process system). In a single-process system, a Python dict with TTL logic is simpler and has zero failure modes. Redis remains the obvious choice if/when multi-process or multi-machine deployment becomes necessary.

## Storage

| Technology | Use Case | Complexity | Backup Story | Query Capability |
|-----------|----------|------------|--------------|-----------------|
| **SQLite (WAL mode)** | All persistent memory | Zero (one file) | Copy file | Full SQL + FTS5 |
| PostgreSQL + pgvector | Memory at scale | High (separate process) | pg_dump | Full SQL + vector ops |
| Redis | Working memory / cache | Medium | RDB/AOF | Key-value only |

**Current implementation: SQLite for everything persistent.** See Memory Architecture (§E) for full rationale.

---

# §K — Repository Structure

The repository as it is now. The original Phase 0 plan (with `vision/`, `screen/`,
`state/`, `output/`, `tools/`, `privacy/` and `observability/` packages) is kept in
`docs/superpowers/specs/2026-08-08-jarvis-architecture-design.md`; those packages
were not built.

```
jarvis/
├── app/
│   ├── main.py                    # Entry point, lifecycle, wiring
│   ├── config.py                  # Dataclasses loaded from configs/default.toml
│   ├── contracts.py               # Turn, Transcript, latency stages, LatencyRecorder
│   ├── audio/
│   │   ├── capture.py             # Microphone capture (sounddevice callback + bounded ring buffer)
│   │   ├── vad.py                 # Silero VAD wrapper
│   │   ├── segmentation.py        # Endpointing / speech segmenter
│   │   ├── asr.py                 # Granite Speech 5 via mlx-audio
│   │   └── frontend.py            # AudioFrontend: capture → VAD → ASR → Transcript, barge-in
│   ├── attention/                 # engine.py, rules.py (tier 1), classifier.py (tier 2 scorer)
│   ├── events/                    # bus.py (asyncio pub/sub), types.py (event dataclasses)
│   ├── context/builder.py         # Context assembly for the LLM
│   ├── memory/                    # working.py, conversation.py, manager.py, sqlite_store.py (FTS5), store.py, types.py
│   ├── reasoning/
│   │   ├── model.py               # ReasoningModel / StreamingReasoningModel protocols
│   │   ├── local_model.py         # Streaming Ollama client
│   │   └── fallback.py            # Primary → local fallback wrapper
│   ├── response/decision.py       # Respond-or-not decision
│   ├── tts/
│   │   ├── model.py               # TextToSpeech / StreamingTextToSpeech / StreamingAudioPlayer protocols
│   │   ├── local_tts.py           # Phonon TTS via ptts
│   │   └── player.py              # LocalAudioPlayer, StreamingPlayer (bounded, turn-tagged)
│   ├── perception/                # camera.py, sampler.py, local_vision.py (Ollama vision), pipeline.py, visual.py, types.py
│   ├── fusion/engine.py           # Multimodal context fusion
│   └── orchestration/turns.py     # TurnManager (being integrated; not yet in the tree)
├── configs/
│   ├── default.toml               # All configuration
│   └── system_prompt.txt
├── scripts/                       # benchmark_*.py, stability_test.py, e2e_fakes.py (see 07-performance.md)
├── tests/                         # one test_<package>/ per app package, plus test_config/ and test_scripts/
├── docs/architecture/             # This document
└── pyproject.toml
```

The SQLite database is created at runtime at `memory.database_path` (`~/.jarvis/jarvis.db`).

---

# §L — Interface Definitions (Conceptual Level)

```python
"""
Typed pseudocode for all major inter-component interfaces.
Not implementations — contracts that allow independent development and testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator


# ─── Enums ─────────────────────────────────────────────────────────────

class Modality(Enum):
    AUDIO = auto()
    VISUAL = auto()
    SCREEN = auto()
    SYSTEM = auto()

class Decision(Enum):
    """ResponseDecision action tags (parsed from LLM output)."""
    RESPOND = auto()
    SILENT = auto()
    STORE = auto()
    EXECUTE_TOOL = auto()
    ASK_CLARIFICATION = auto()

class PermissionLevel(Enum):
    SAFE = auto()        # No confirmation needed (read-only, informational)
    MODERATE = auto()    # Confirm first time, remember preference
    DANGEROUS = auto()   # Always confirm (file deletion, system commands)


# ─── Events ────────────────────────────────────────────────────────────

# Canonical event envelope — the contract used by every subsystem.
# All perception events extend this base.

@dataclass
class Event:
    """Base event envelope. Every event in the system carries these fields."""
    id: str                     # Uniquely identifies this event (UUID)
    timestamp: float            # When the event occurred (Unix timestamp)
    session_id: str             # Current JARVIS runtime/session identifier
    correlation_id: str | None  # Cross-modal grouping (None if uncorrelated)
    source: Modality            # Originating modality
    confidence: float           # Perception confidence (0.0 - 1.0)

@dataclass
class SpeechEvent(Event):
    """Phase 1: speech perception event."""
    transcript: str = ""
    duration_seconds: float = 0.0
    is_question: bool = False

@dataclass
class VisualEvent(Event):
    """Phase 7+: visual perception event."""
    description: str = ""
    objects: list[str] = field(default_factory=list)
    scene_change_magnitude: float = 0.0

@dataclass
class ScreenEvent(Event):
    """Phase 9+: screen perception event."""
    app_name: str = ""
    window_title: str = ""
    visible_text: str = ""
    change_type: str = ""

@dataclass
class SystemEvent(Event):
    """Internal system lifecycle events (startup, shutdown, errors)."""
    event_type: str = ""
    detail: str = ""


# ─── Attention ──────────────────────────────────────────────────────────

@dataclass
class AttentionDecision:
    should_reason: bool
    salience: float          # 0.0 - 1.0
    relevance: float         # how relevant to current context
    urgency: float           # 0.0 - 1.0
    reason: str
    tier_decided: int        # 1, 2, or 3

class AttentionEngine(ABC):
    @abstractmethod
    async def evaluate(self, event: Event) -> AttentionDecision:
        ...


# ─── World State ────────────────────────────────────────────────────────

class WorldStateManager(ABC):
    @abstractmethod
    def update(self, event: Event) -> None:
        ...

    @abstractmethod
    def get_state(self) -> 'WorldState':
        ...


# ─── Memory ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    id: str
    content: str
    importance: float
    timestamp: float
    source: str
    embedding: list[float] | None = None

@dataclass
class MemoryQuery:
    text: str
    max_results: int = 5
    min_importance: float = 0.0
    recency_bias: float = 0.3  # 0 = no bias, 1 = only recent

class MemoryStore(ABC):
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        ...

    @abstractmethod
    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        ...

    @abstractmethod
    async def update(self, id: str, **fields) -> None:
        ...

    @abstractmethod
    async def forget(self, id: str) -> None:
        ...


# ─── Context ────────────────────────────────────────────────────────────

@dataclass
class AssembledContext:
    system_prompt: str
    user_message: str
    memories: list[str]
    environment: str        # Current activity, screen, visual
    token_count: int        # Estimated

class ContextBuilder(ABC):
    @abstractmethod
    async def build(self, event: Event, attention: AttentionDecision, world: 'WorldState') -> AssembledContext:
        ...


# ─── Models ─────────────────────────────────────────────────────────────

class ReasoningModel(ABC):
    @abstractmethod
    async def generate(self, context: AssembledContext) -> 'ReasoningResult':
        ...

@dataclass
class ReasoningResult:
    text: str
    decision: Decision
    confidence: float
    tool_call: dict | None = None    # If decision == EXECUTE_TOOL
    memory_update: dict | None = None # If decision == STORE
    latency_ms: float = 0.0
    tokens_used: int = 0

@dataclass
class ResponseDecision:
    respond: bool
    store: bool
    execute_tool: bool
    clarification_required: bool
    reason: str
    tool_call: dict | None = None

class SpeechRecognizer(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        ...

class VoiceActivityDetector(ABC):
    @abstractmethod
    def process_chunk(self, chunk: bytes) -> bool:
        ...

class TextToSpeech(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        ...

class EmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...


# ─── Tools ──────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None

class Tool(ABC):
    name: str
    description: str
    permission_level: PermissionLevel

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...

class ToolRegistry(ABC):
    @abstractmethod
    def register(self, tool: Tool) -> None:
        ...

    @abstractmethod
    def get(self, name: str) -> Tool | None:
        ...

    @abstractmethod
    def list_tools(self) -> list[Tool]:
        ...


# ─── Event Bus ──────────────────────────────────────────────────────────

from typing import Callable, Awaitable

EventHandler = Callable[[Event], Awaitable[None]]

class EventBus(ABC):
    """
    In-process asyncio event bus with bounded queues.

    Backpressure policy:
    - Each subscriber has a bounded asyncio.Queue (default capacity: 256 events).
    - Audio capture must NEVER block because a consumer is slow.
    - publish() is non-blocking: if a subscriber queue is full, the oldest
      event in that queue is dropped and a warning is logged.
    - Finalized SpeechEvents must not be silently lost under normal operation.
      Queue saturation is logged at WARNING level with event id and subscriber name.
    - The bounded queue prevents unbounded memory growth if a consumer stalls.

    Overflow behavior (Phase 1):
      Producer faster than consumer → bounded queue → drop oldest + log warning
    """

    @abstractmethod
    async def publish(self, event: Event) -> None:
        ...

    @abstractmethod
    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        ...


# ─── Privacy ────────────────────────────────────────────────────────────

class PrivacyController(ABC):
    @abstractmethod
    def is_perception_allowed(self, modality: Modality) -> bool:
        ...

    @abstractmethod
    def enter_privacy_mode(self) -> None:
        ...

    @abstractmethod
    def exit_privacy_mode(self) -> None:
        ...

    @abstractmethod
    def get_active_permissions(self) -> dict[Modality, bool]:
        ...


# ─── Observability ──────────────────────────────────────────────────────

@dataclass
class MetricEvent:
    name: str
    value: float
    unit: str
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)

class MetricsCollector(ABC):
    @abstractmethod
    def record(self, metric: MetricEvent) -> None:
        ...

    @abstractmethod
    def get_summary(self, name: str, window_seconds: int = 60) -> dict:
        ...
```
