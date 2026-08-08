"""Perception types — Frame, VisualAnalysis, visual change detection primitives.

Frame is the internal representation of a captured image.
VisualAnalysis is the output of a VisualAnalyzer.
These types are decoupled from any particular capture backend or vision model.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto


class VisualEventType(Enum):
    SCENE_CHANGE = auto()
    PERSON_DETECTED = auto()
    OBJECT_DETECTED = auto()
    TEXT_DETECTED = auto()
    VISUAL_SUMMARY = auto()


@dataclass(frozen=True)
class Frame:
    """A single captured image, decoupled from the capture backend."""

    timestamp: float
    data: bytes
    width: int
    height: int
    format: str  # e.g. "rgb24", "bgr24", "jpeg"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualAnalysis:
    """Output of a VisualAnalyzer — structured scene understanding."""

    timestamp: float
    description: str
    event_type: VisualEventType
    detections: tuple[Detection, ...] = ()
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)
