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
class LoggingConfig:
    level: str = "INFO"


@dataclass
class JarvisConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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
    if "logging" in raw:
        cfg.logging = LoggingConfig(**{k: v for k, v in raw["logging"].items() if k in LoggingConfig.__dataclass_fields__})
    return cfg
