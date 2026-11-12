from __future__ import annotations

from app.config import ReasoningConfig
from app.reasoning.fallback import FallbackModel
from app.reasoning.local_model import OllamaModel
from app.reasoning.model import StreamingReasoningModel


def build_reasoning_model(cfg: ReasoningConfig) -> StreamingReasoningModel:
    """The primary Ollama model, wrapped with a local fallback when cfg.fallback_model is set."""
    common = {
        "timeout_seconds": cfg.timeout_seconds,
        "connect_timeout_seconds": cfg.connect_timeout_seconds,
        "keep_alive": cfg.keep_alive,
        "think": cfg.think,
    }
    primary = OllamaModel(model=cfg.model, base_url=cfg.ollama_url, **common)
    if not cfg.fallback_model:
        return primary
    secondary = OllamaModel(model=cfg.fallback_model, base_url=cfg.fallback_url, **common)
    return FallbackModel(primary, secondary, cooldown_seconds=cfg.fallback_cooldown_seconds)
