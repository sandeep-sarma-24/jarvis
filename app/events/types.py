from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto


class Modality(Enum):
    AUDIO = auto()
    VISUAL = auto()
    SCREEN = auto()
    SYSTEM = auto()


@dataclass
class Event:
    """Canonical event envelope — contract used by every subsystem."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    correlation_id: str | None = None
    source: Modality = Modality.SYSTEM
    confidence: float = 0.0


@dataclass
class SpeechEvent(Event):
    """Phase 1 — audio perception produced a transcript."""

    transcript: str = ""
    duration_seconds: float = 0.0
    is_question: bool = False  # kept for arch compat, not used in Phase 1
    metadata: dict[str, object] = field(default_factory=dict)
