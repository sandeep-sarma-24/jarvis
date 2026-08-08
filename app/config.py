from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.toml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    ring_buffer_seconds: int = 30
    chunk_samples: int = 512


@dataclass
class VADConfig:
    threshold: float = 0.5
    speech_end_silence_ms: int = 300
    minimum_speech_ms: int = 250
    maximum_segment_seconds: int = 30


@dataclass
class ASRConfig:
    model: str = "Qwen/Qwen3-ASR-0.6B"


@dataclass
class EventBusConfig:
    subscriber_queue_size: int = 256


@dataclass
class AttentionConfig:
    enabled: bool = True
    reason_threshold: float = 0.4
    assistant_names: list[str] = field(default_factory=lambda: ["jarvis"])
    working_memory_size: int = 20


@dataclass
class ReasoningConfig:
    enabled: bool = True
    model: str = "gemma4:12b"
    ollama_url: str = "http://localhost:11434"
    max_context_messages: int = 8
    max_context_characters: int = 6000
    max_output_tokens: int = 256
    temperature: float = 0.2
    timeout_seconds: float = 30.0
    max_concurrent_requests: int = 1


@dataclass
class ResponseConfig:
    enabled: bool = True


@dataclass
class TTSConfig:
    enabled: bool = True
    model: str = "mlx-community/Kokoro-82M-bf16"
    voice: str = "bf_emma"
    speed: float = 1.0
    sample_rate: int = 24000
    max_text_length: int = 2000


@dataclass
class PlaybackConfig:
    enabled: bool = True
    volume: float = 1.0


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class MemoryConfig:
    enabled: bool = True
    database_path: str = "~/.jarvis/jarvis.db"
    max_retrieved_memories: int = 5
    max_recent_messages: int = 8


@dataclass
class JarvisConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_config(path: Path | None = None) -> JarvisConfig:
    path = path or _DEFAULT_CONFIG
    if not path.exists():
        return JarvisConfig()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = JarvisConfig()
    if "audio" in raw:
        cfg.audio = AudioConfig(**{k: v for k, v in raw["audio"].items() if k in AudioConfig.__dataclass_fields__})
    if "vad" in raw:
        cfg.vad = VADConfig(**{k: v for k, v in raw["vad"].items() if k in VADConfig.__dataclass_fields__})
    if "asr" in raw:
        cfg.asr = ASRConfig(**{k: v for k, v in raw["asr"].items() if k in ASRConfig.__dataclass_fields__})
    if "event_bus" in raw:
        cfg.event_bus = EventBusConfig(**{k: v for k, v in raw["event_bus"].items() if k in EventBusConfig.__dataclass_fields__})
    if "attention" in raw:
        cfg.attention = AttentionConfig(**{k: v for k, v in raw["attention"].items() if k in AttentionConfig.__dataclass_fields__})
    if "reasoning" in raw:
        cfg.reasoning = ReasoningConfig(**{k: v for k, v in raw["reasoning"].items() if k in ReasoningConfig.__dataclass_fields__})
    if "response" in raw:
        cfg.response = ResponseConfig(**{k: v for k, v in raw["response"].items() if k in ResponseConfig.__dataclass_fields__})
    if "tts" in raw:
        cfg.tts = TTSConfig(**{k: v for k, v in raw["tts"].items() if k in TTSConfig.__dataclass_fields__})
    if "playback" in raw:
        cfg.playback = PlaybackConfig(**{k: v for k, v in raw["playback"].items() if k in PlaybackConfig.__dataclass_fields__})
    if "logging" in raw:
        cfg.logging = LoggingConfig(**{k: v for k, v in raw["logging"].items() if k in LoggingConfig.__dataclass_fields__})
    if "memory" in raw:
        cfg.memory = MemoryConfig(**{k: v for k, v in raw["memory"].items() if k in MemoryConfig.__dataclass_fields__})
    return cfg
