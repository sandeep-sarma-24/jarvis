# §J — Technology Comparison and Recommendations

## Voice Activity Detection (VAD)

| Technology | F1 Score | Latency | Memory | License | Apple Silicon | Streaming | Recommendation |
|-----------|----------|---------|--------|---------|--------------|-----------|----------------|
| **Silero VAD v5** | 95.95% | <1ms/chunk (ONNX) | ~50MB | MIT | Yes (CPU) | Yes | **Recommended for V1** |
| FireRedVAD | 97.57% | ~10ms/frame | ~100MB | Apache 2.0 | Via ONNX/NCNN | Yes | Better accuracy, evaluate for V2 |
| WebRTC VAD | 52.30% | <0.1ms | <1MB | BSD | Yes | Yes | Too inaccurate |

**Decision: Silero VAD v5.** Most mature, best documented, <1ms on CPU, proven Python integration. FireRedVAD is more accurate (97.57% vs 95.95%) but newer (March 2026) and less ecosystem integration. Upgrade path: swap Silero → FireRedVAD behind the `VoiceActivityDetector` interface once it matures.

Sources: [Silero VAD GitHub](https://github.com/snakers4/silero-vad), [FireRedVAD HuggingFace](https://huggingface.co/FireRedTeam/FireRedVAD)

## Speech Recognition (ASR)

| Technology | WER (LibriSpeech clean) | Latency (M4) | Memory | Languages | License | MLX Native |
|-----------|------------------------|--------------|--------|-----------|---------|-----------|
| **Qwen3-ASR 0.6B** | 2.29% (fp16) | ~460ms (short clip) | 1.2GB | 30+ | Apache 2.0 | Yes (mlx-qwen3-asr) |
| Qwen3-ASR 1.7B | 1.99% | ~4s | 3.4GB | 30+ | Apache 2.0 | Yes |
| Whisper Large v3 | 2.11% | ~2-3s | 3GB | 100+ | MIT | Via whisper.cpp |
| Parakeet TDT | ~2.1% | <500ms | 634MB (CoreML) | 25 | Apache 2.0 | CoreML (Neural Engine) |

**Decision: Qwen3-ASR 0.6B via mlx-qwen3-asr.** Best quality-to-latency ratio. 460ms for a short clip on M4 is well within our 800ms budget. 1.2GB memory is manageable. Native MLX implementation means no PyTorch dependency for inference. Falls back to 1.7B if accuracy is insufficient for the user's accent/environment.

Sources: [mlx-qwen3-asr benchmarks](https://github.com/moona3k/mlx-qwen3-asr/blob/main/docs/BENCHMARKS.md), [Qwen3-ASR HuggingFace](https://huggingface.co/collections/Qwen/qwen3-asr)

## Text-to-Speech (TTS)

| Technology | Quality | Latency | Memory | Voices | License | MLX Native | Voice Character |
|-----------|---------|---------|--------|--------|---------|-----------|----------------|
| **Kokoro-82M** | 9.2/10 | 3-5x RT (CPU), ~45ms (CoreML) | ~500MB | 54 | Apache 2.0 | Yes (kokoro-mlx) | Fixed presets (BF_Emma = British female) |
| Qwen3-TTS 0.6B | High | ~120ms TTFA | ~2.3GB (bf16) | Voice cloning | Apache 2.0 | Yes (Holler) | Customizable |
| Holler (Qwen3-TTS fine-tune) | High | 120ms TTFA, 2.5x RTF | 1.7GB (6-bit) | 6 American | Apache 2.0 | Yes | American English, streaming |

**Decision: Kokoro-82M via kokoro-mlx for V1.** Smallest, fastest, most mature MLX integration. BF_Emma voice provides British inflection appropriate for JARVIS personality. Upgrade path: Qwen3-TTS/Holler for voice cloning and richer prosody once the base system works.

**JARVIS voice note:** Kokoro's `bf_emma` voice is the closest match to the MCU JARVIS tone (British, composed). For a male British voice, Kokoro's selection is more limited. If no satisfactory male British voice exists in Kokoro's 54-voice library, Qwen3-TTS with voice cloning becomes the V2 upgrade path.

Sources: [kokoro-mlx PyPI](https://pypi.org/project/kokoro-mlx/), [Holler GitHub](https://github.com/sentiuminc/holler)

## Primary LLM (Reasoning)

| Technology | Multimodal | Size (Q4) | Context | tok/s (M4 Air est.) | License |
|-----------|-----------|-----------|---------|---------------------|---------|
| **Gemma 4 12B Unified** | Text+Image+Audio | 7.6GB | 256K | ~25-35 | Apache 2.0 |
| Llama 3.1 8B | Text only | 4.5GB | 128K | ~45-55 | Llama |
| Qwen3 14B | Text only | 8.5GB | 128K | ~20-25 | Apache 2.0 |
| Phi-4 14B | Text+Image | 8.5GB | 16K | ~20-25 | MIT |

**Decision: Gemma 4 12B Unified.** Only option with native text+image+audio in a single model at this size class. This eliminates separate vision/audio models for the reasoning tier. The encoder-free architecture means lower multimodal latency. Context management is critical at 16GB — keep prompts under 4K tokens.

Sources: [Google AI Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4), [AI on Mac benchmarks](https://ai-on-mac.com/articles/gemma-4-12b-mac-en/)

## Inference Runtime

| Runtime | tok/s (12B, M4 Air est.) | Multi-model | API | Maturity | Cross-platform |
|---------|--------------------------|-------------|-----|----------|---------------|
| **Ollama** | ~25-30 | Yes | OpenAI-compat | High | Mac/Linux/Win |
| llama.cpp (raw) | ~28-33 | Manual | OpenAI-compat | High | Mac/Linux/Win |
| MLX (mlx-lm) | ~30-38 | Limited | Python | Medium | Mac only |
| LiteRT-LM | ~35-45 (unconfirmed on Air) | Yes | OpenAI-compat | Low (v0.13) | Mac/Linux/Win |

**Decision: Ollama for LLM serving.** Best balance of throughput, multi-model support, API stability, and developer experience. The ~10% throughput sacrifice vs raw MLX is worth the operational simplicity. LiteRT-LM is the upgrade candidate once confirmed on base M4.

Sources: [Contra Collective comparison](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026), [Ollama docs](https://ollama.ai)

## Embedding Model

| Model | Dimensions | Size | Latency | Quality (MTEB) |
|-------|-----------|------|---------|----------------|
| **nomic-embed-text** | 768 | ~500MB | <10ms | High |
| all-MiniLM-L6-v2 | 384 | ~80MB | <5ms | Medium |
| mxbai-embed-large | 1024 | ~1.2GB | ~15ms | Very High |

**Decision: nomic-embed-text via Ollama.** Good quality, reasonable size, available through the same Ollama interface as the LLM. The 768-dim embeddings provide good retrieval quality for memory search.

## Event Bus / IPC

| Technology | Latency | Persistence | Complexity | Operational Familiarity |
|-----------|---------|-------------|------------|------------------------|
| **asyncio queues (in-process)** | <0.01ms | None (in-memory) | Zero | N/A (stdlib) |
| NATS | ~1ms (local) | Optional (JetStream) | Medium (separate process) | High (author's day job) |
| Redis Pub/Sub | ~0.5ms | No (fire-and-forget) | Low-Medium (separate process) | High (author's day job) |
| ZeroMQ | ~0.1ms | No | Low | Low |

**Decision: asyncio queues for V1.** For a single-process Python monolith, in-process async queues are the correct abstraction. Zero infrastructure, zero latency, zero failure modes. The event bus is just a typed async queue with subscriber registration.

**When to reconsider NATS:** If the system splits into multiple processes (e.g., a separate perception daemon), NATS becomes the obvious upgrade. The interface (`publish(event)` / `subscribe(pattern, handler)`) is the same shape either way — only the transport changes.

**Redis evaluation:** Redis adds value when you need cross-process shared state with TTLs (exactly what working memory would be in a multi-process system). In a single-process system, a Python dict with TTL logic is simpler and has zero failure modes. Redis remains the obvious choice if/when multi-process or multi-machine deployment becomes necessary.

## Storage

| Technology | Use Case | Complexity | Backup Story | Query Capability |
|-----------|----------|------------|--------------|-----------------|
| **SQLite (WAL mode)** | All persistent memory | Zero (one file) | Copy file | Full SQL + FTS5 |
| PostgreSQL + pgvector | Memory at scale | High (separate process) | pg_dump | Full SQL + vector ops |
| Redis | Working memory / cache | Medium | RDB/AOF | Key-value only |

**Decision: SQLite for everything persistent.** See Memory Architecture (§E) for full rationale.

---

# §K — Repository Structure

```
jarvis/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Entry point, lifecycle management
│   ├── config.py                  # Configuration loading (env/file)
│   │
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py             # Microphone capture (pyaudio/sounddevice)
│   │   ├── vad.py                 # VAD wrapper (Silero)
│   │   └── asr.py                 # ASR wrapper (Qwen3-ASR)
│   │
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── capture.py             # Camera capture (OpenCV/AVFoundation)
│   │   ├── diff.py                # Frame differencing
│   │   └── describe.py            # Vision description (Gemma 4 multimodal)
│   │
│   ├── screen/
│   │   ├── __init__.py
│   │   ├── capture.py             # Screen capture (CGWindowList)
│   │   ├── diff.py                # Screen change detection (pHash)
│   │   ├── ocr.py                 # OCR (Apple Vision framework)
│   │   └── app_id.py              # App identification (Accessibility API)
│   │
│   ├── events/
│   │   ├── __init__.py
│   │   ├── bus.py                 # Event bus (asyncio-based pub/sub)
│   │   └── types.py               # Event dataclasses (PerceptionEvent, etc.)
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── engine.py              # Tiered attention orchestrator
│   │   ├── rules.py               # Tier 1: deterministic rules
│   │   ├── classifier.py          # Tier 2: lightweight classifier
│   │   └── llm_judge.py           # Tier 3: LLM judgment (rare)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working.py             # Working memory (in-process state)
│   │   ├── episodic.py            # Episodic memory (SQLite)
│   │   ├── semantic.py            # Semantic memory (SQLite)
│   │   ├── procedural.py          # Procedural memory (SQLite)
│   │   ├── embeddings.py          # Embedding operations
│   │   ├── retrieval.py           # Unified retrieval across layers
│   │   └── lifecycle.py           # Decay, summarization, deduplication
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   └── builder.py             # Context assembly for LLM
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── model.py               # LLM interface + Ollama client
│   │   └── decision.py            # Decision parsing (respond/silent/store/tool)
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   ├── tts.py                 # TTS wrapper (Kokoro)
│   │   └── speaker.py             # Audio playback
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py            # Tool registry
│   │   └── base.py                # Tool base class
│   │
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── controller.py          # Privacy mode, permissions
│   │   └── indicators.py          # Active perception indicators
│   │
│   └── observability/
│       ├── __init__.py
│       ├── logging.py             # Structured logging
│       └── metrics.py             # Performance metrics collection
│
├── tests/
│   ├── __init__.py
│   ├── test_attention/
│   ├── test_memory/
│   ├── test_audio/
│   ├── test_context/
│   └── conftest.py                # Shared fixtures
│
├── scripts/
│   ├── setup_models.py            # Download/configure required models
│   └── benchmark.py               # Latency benchmarking
│
├── configs/
│   ├── default.toml               # Default configuration
│   └── models.toml                # Model configurations per hardware
│
├── data/
│   └── jarvis.db                  # SQLite database (created at runtime)
│
├── docs/
│   └── architecture/              # This document
│
├── pyproject.toml                  # Project config + dependencies
├── README.md
└── .gitignore
```

**Deviation from brief's suggested structure:** Merged `perception/` into `audio/`, `vision/`, `screen/` because each modality has its own capture + processing logic that doesn't share code. Added `context/` as a first-class module (it's load-bearing enough to warrant it). Added `output/` to group TTS + playback. Removed `models/` directory (models are managed by Ollama, not checked into the repo). Removed `api/` (no external API in V1 — this is a local system).

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

class Decision(Enum):
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

@dataclass
class PerceptionEvent:
    id: str
    timestamp: float
    source: Modality
    importance_hint: float = 0.5

@dataclass
class SpeechEvent(PerceptionEvent):
    transcript: str = ""
    confidence: float = 0.0
    duration_seconds: float = 0.0
    is_question: bool = False

@dataclass
class VisualEvent(PerceptionEvent):
    description: str = ""
    objects: list[str] = field(default_factory=list)
    scene_change_magnitude: float = 0.0

@dataclass
class ScreenEvent(PerceptionEvent):
    app_name: str = ""
    window_title: str = ""
    visible_text: str = ""
    change_type: str = ""


# ─── Attention ──────────────────────────────────────────────────────────

@dataclass
class AttentionScore:
    should_reason: bool
    salience: float          # 0.0 - 1.0
    addressed_to_system: float
    urgency: float
    reason: str
    tier_decided: int        # 1, 2, or 3

class AttentionEngine(ABC):
    @abstractmethod
    async def evaluate(self, event: PerceptionEvent) -> AttentionScore:
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
    async def build(self, event: PerceptionEvent, attention: AttentionScore) -> AssembledContext:
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

EventHandler = Callable[[PerceptionEvent], Awaitable[None]]

class EventBus(ABC):
    @abstractmethod
    async def publish(self, event: PerceptionEvent) -> None:
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
