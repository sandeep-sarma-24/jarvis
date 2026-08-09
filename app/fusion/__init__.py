"""Multimodal temporal fusion — Phase 8.

Combines recent audio and visual observations into bounded temporal snapshots
for context assembly. Never invokes an LLM. Never blocks capture pipelines.
"""

from app.fusion.engine import MultimodalFusion, MultimodalSnapshot

__all__ = ["MultimodalFusion", "MultimodalSnapshot"]
