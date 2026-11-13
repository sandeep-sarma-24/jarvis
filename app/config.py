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
    speech_end_silence_ms: int = 192
    minimum_speech_ms: int = 250
    maximum_segment_seconds: int = 30
    pre_roll_ms: int = 96  # audio kept from before speech onset
    short_utterance_ms: int = 1000  # below this much speech, use the longer silence below
    short_utterance_silence_ms: int = 480


@dataclass
class ASRConfig:
    model: str = "ibm-granite/granite-speech-5.0-470m-turboctc"


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
    connect_timeout_seconds: float = 2.0
    keep_alive: str = "30m"
    think: bool = False
    fallback_url: str = "http://127.0.0.1:11434"
    fallback_model: str = ""  # empty = no fallback backend
    fallback_cooldown_seconds: float = 30.0


@dataclass
class ResponseConfig:
    enabled: bool = True


@dataclass
class TTSConfig:
    enabled: bool = True
    model: str = "hf://gradium/phonon-31m-en"
    voice: str = "Easton"
    quant: str = "q8_0"
    max_text_length: int = 2000


@dataclass
class PlaybackConfig:
    enabled: bool = True
    volume: float = 1.0
    max_queue_seconds: float = 2.0  # bound on synthesized-but-unplayed audio
    tail_ms: int = 200  # keep the mic suppressed this long after playback ends


@dataclass
class BargeInConfig:
    enabled: bool = False  # needs headphones or echo cancellation; see docs
    vad_threshold: float = 0.8
    min_speech_ms: int = 256


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class MemoryConfig:
    enabled: bool = True
    database_path: str = "~/.jarvis/jarvis.db"
    max_retrieved_memories: int = 5
    max_recent_messages: int = 8
    cross_session: bool = True
    semantic_enabled: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"


@dataclass
class CameraConfig:
    enabled: bool = False  # disabled by default — privacy
    device_index: int = 0
    width: int = 640
    height: int = 480
    capture_fps: float = 10.0
    analysis_fps: float = 1.0
    max_frame_age_seconds: float = 2.0
    max_queue_size: int = 2
    change_threshold: float = 0.05
    dedup_window_seconds: float = 30.0


@dataclass
class VisionConfig:
    enabled: bool = False  # disabled by default — no vision model requirement
    provider: str = "ollama"
    model: str = "gemma4:12b"
    ollama_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    max_image_size: int = 1024
    temperature: float = 0.1
    prompt: str = ""  # empty = use default prompt


@dataclass
class PerceptionConfig:
    enabled: bool = False  # disabled by default — privacy
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    max_visual_events: int = 5
    max_visual_context_characters: int = 500
    max_visual_event_age_seconds: float = 60.0
    on_demand_timeout_seconds: float = 3.0
    pause_during_turn: bool = True


@dataclass
class FusionConfig:
    enabled: bool = True
    window_seconds: float = 15.0
    max_events: int = 20
    max_characters: int = 3000


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
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


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
    if "barge_in" in raw:
        cfg.barge_in = BargeInConfig(**{k: v for k, v in raw["barge_in"].items() if k in BargeInConfig.__dataclass_fields__})
    if "logging" in raw:
        cfg.logging = LoggingConfig(**{k: v for k, v in raw["logging"].items() if k in LoggingConfig.__dataclass_fields__})
    if "memory" in raw:
        cfg.memory = MemoryConfig(**{k: v for k, v in raw["memory"].items() if k in MemoryConfig.__dataclass_fields__})
    if "perception" in raw:
        p = raw["perception"]
        nested = {"camera", "vision"}
        top = {k: v for k, v in p.items() if k in PerceptionConfig.__dataclass_fields__ and k not in nested}
        cam = CameraConfig()
        if "camera" in p:
            cam = CameraConfig(**{k: v for k, v in p["camera"].items() if k in CameraConfig.__dataclass_fields__})
        vis = VisionConfig()
        if "vision" in p:
            vis = VisionConfig(**{k: v for k, v in p["vision"].items() if k in VisionConfig.__dataclass_fields__})
        cfg.perception = PerceptionConfig(camera=cam, vision=vis, **top)
    if "fusion" in raw:
        cfg.fusion = FusionConfig(**{k: v for k, v in raw["fusion"].items() if k in FusionConfig.__dataclass_fields__})
    return cfg
