"""Tier 2 — Deterministic weighted scorer.

Extracts features from a SpeechEvent and computes a salience score
via weighted combination. No ML model — just arithmetic.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.attention.rules import detect_question, detect_request


@dataclass
class ScorerWeights:
    """Configurable weights for salience scoring.

    ponytail: these are initial heuristic values, not empirically optimized.
    Upgrade path: tune from labeled attention_cases.json data.
    """
    explicit_address: float = 0.50
    question: float = 0.40
    request_language: float = 0.30
    recent_interaction: float = 0.10
    technical_terms: float = 0.15
    transcript_length: float = 0.05
    background_signal: float = -0.30


_TECH_TERMS = re.compile(
    r"\b(?:api|error|bug|deploy|server|database|function|class|module"
    r"|variable|exception|stack\s*trace|debug|commit|merge|branch|repo"
    r"|container|docker|kubernetes|endpoint|request|response|http|sql"
    r"|query|schema|migration|test|lint|build|compile|runtime|crash"
    r"|memory\s*leak|segfault|null|undefined|import|package|dependency"
    r"|config|env|token|auth|permission|cors|ssl|dns|port|socket"
    r"|regex|parse|serialize|async|await|thread|process|cpu|gpu)\b",
    re.I,
)

_BACKGROUND_SIGNALS = re.compile(
    r"\b(?:i'?ll call you|see you later|talk to you|bye|goodbye"
    r"|pass me|hand me|over there|let'?s go|come on"
    r"|where did i put|where are my)\b",
    re.I,
)


@dataclass
class ScorerFeatures:
    explicit_address: float = 0.0
    question: float = 0.0
    request_language: float = 0.0
    recent_interaction: float = 0.0
    technical_terms: float = 0.0
    transcript_length: float = 0.0
    background_signal: float = 0.0


class AttentionScorer:
    """Tier 2 weighted feature scorer."""

    def __init__(self, weights: ScorerWeights | None = None) -> None:
        self._w = weights or ScorerWeights()

    def score(
        self,
        transcript: str,
        *,
        is_addressed: bool = False,
        last_interaction_age_seconds: float | None = None,
    ) -> tuple[float, ScorerFeatures]:
        """Compute salience score and return (score, features).

        Returns salience clamped to [0.0, 1.0].
        """
        features = self._extract_features(
            transcript,
            is_addressed=is_addressed,
            last_interaction_age_seconds=last_interaction_age_seconds,
        )

        raw = (
            features.explicit_address * self._w.explicit_address
            + features.question * self._w.question
            + features.request_language * self._w.request_language
            + features.recent_interaction * self._w.recent_interaction
            + features.technical_terms * self._w.technical_terms
            + features.transcript_length * self._w.transcript_length
            + features.background_signal * self._w.background_signal
        )

        return max(0.0, min(1.0, raw)), features

    def _extract_features(
        self,
        transcript: str,
        *,
        is_addressed: bool,
        last_interaction_age_seconds: float | None,
    ) -> ScorerFeatures:
        text = transcript.strip()
        f = ScorerFeatures()

        f.explicit_address = 1.0 if is_addressed else 0.0
        f.question = detect_question(text)
        f.request_language = detect_request(text)

        if last_interaction_age_seconds is not None and last_interaction_age_seconds < 30.0:
            f.recent_interaction = max(0.0, 1.0 - last_interaction_age_seconds / 30.0)

        if text:
            tech_matches = len(_TECH_TERMS.findall(text))
            f.technical_terms = min(1.0, tech_matches / 3.0)

        word_count = len(text.split()) if text else 0
        if word_count >= 3:
            f.transcript_length = min(1.0, word_count / 15.0)

        if _BACKGROUND_SIGNALS.search(text):
            f.background_signal = 1.0

        return f
