"""Attention Engine — tiered evaluation orchestrator.

SpeechEvent → Tier 1 rules → (if undecided) Tier 2 scorer → AttentionDecision.
Never raises on classification failure; fails closed (should_reason=False).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.attention.classifier import AttentionScorer, ScorerWeights
from app.attention.rules import AttentionRules
from app.events.types import SpeechEvent, VisualEvent
from app.memory.working import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionDecision:
    should_reason: bool
    salience: float
    reason: str
    tier: str


_FALLBACK = AttentionDecision(
    should_reason=False, salience=0.0,
    reason="attention evaluation failed", tier="fallback",
)


class AttentionEngine:
    """Tiered attention: rules → scorer → threshold → decision."""

    def __init__(
        self,
        *,
        assistant_names: list[str] | None = None,
        reason_threshold: float = 0.5,
        scorer_weights: ScorerWeights | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self._rules = AttentionRules(assistant_names=assistant_names)
        self._scorer = AttentionScorer(weights=scorer_weights)
        self._threshold = reason_threshold
        self._memory = working_memory if working_memory is not None else WorkingMemory()

        self._visual_salience_threshold = 0.7

        self._counters = {
            "evaluations": 0,
            "reason_true": 0,
            "reason_false": 0,
            "rule_decisions": 0,
            "scorer_decisions": 0,
            "fallback_decisions": 0,
            "visual_evaluations": 0,
            "visual_reason_true": 0,
            "visual_reason_false": 0,
        }

    @property
    def counters(self) -> dict[str, int]:
        return dict(self._counters)

    @property
    def working_memory(self) -> WorkingMemory:
        return self._memory

    def evaluate(self, event: SpeechEvent) -> AttentionDecision:
        self._counters["evaluations"] += 1
        try:
            decision = self._evaluate_inner(event)
        except Exception:
            logger.exception("Attention evaluation failed for event %s", event.id)
            decision = _FALLBACK
            self._counters["fallback_decisions"] += 1

        self._memory.record(event.id, event.transcript, decision.should_reason)

        if decision.should_reason:
            self._counters["reason_true"] += 1
        else:
            self._counters["reason_false"] += 1

        logger.info(
            "ATTENTION event_id=%s transcript=%r should_reason=%s "
            "salience=%.2f tier=%s reason=%r",
            event.id[:12], event.transcript[:80],
            decision.should_reason, decision.salience,
            decision.tier, decision.reason,
        )

        return decision

    def _evaluate_inner(self, event: SpeechEvent) -> AttentionDecision:
        transcript = event.transcript.strip()

        if not transcript:
            self._counters["rule_decisions"] += 1
            return AttentionDecision(
                should_reason=False, salience=0.0,
                reason="empty transcript", tier="rule",
            )

        # Tier 1: deterministic rules
        rule_result = self._rules.evaluate(transcript)
        if rule_result is not None:
            self._counters["rule_decisions"] += 1
            return AttentionDecision(
                should_reason=rule_result.should_reason,
                salience=rule_result.salience,
                reason=rule_result.reason,
                tier="rule",
            )

        # Tier 2: weighted scorer
        last_interaction_age = self._memory.seconds_since_last_interaction()

        salience, features = self._scorer.score(
            transcript,
            is_addressed=False,
            last_interaction_age_seconds=last_interaction_age,
        )

        should_reason = salience >= self._threshold
        reason = self._build_scorer_reason(features, salience)

        self._counters["scorer_decisions"] += 1
        return AttentionDecision(
            should_reason=should_reason,
            salience=round(salience, 4),
            reason=reason,
            tier="scorer",
        )

    def evaluate_visual(self, event: VisualEvent) -> AttentionDecision:
        """Evaluate a visual event for attention-worthiness.

        Conservative policy: only high-salience visual events (e.g. person entering)
        warrant attention. Mundane observations are logged but not acted on.
        """
        self._counters["visual_evaluations"] += 1

        _HIGH_SALIENCE_TYPES = {"PERSON_DETECTED", "SCENE_CHANGE"}

        try:
            if event.event_type in _HIGH_SALIENCE_TYPES and event.confidence >= self._visual_salience_threshold:
                salience = event.confidence
                reason = f"visual: {event.event_type} ({event.description[:60]})"
                decision = AttentionDecision(
                    should_reason=True, salience=salience,
                    reason=reason, tier="visual",
                )
            else:
                decision = AttentionDecision(
                    should_reason=False,
                    salience=event.confidence * 0.3,
                    reason=f"visual: low-salience {event.event_type}",
                    tier="visual",
                )
        except Exception:
            logger.exception("Visual attention evaluation failed for event %s", event.id)
            decision = _FALLBACK
            self._counters["fallback_decisions"] += 1

        if decision.should_reason:
            self._counters["visual_reason_true"] += 1
        else:
            self._counters["visual_reason_false"] += 1

        logger.info(
            "ATTENTION_VISUAL event_id=%s type=%s should_reason=%s salience=%.2f",
            event.id[:12], event.event_type, decision.should_reason, decision.salience,
        )

        return decision

    def _build_scorer_reason(self, features, salience: float) -> str:
        signals = []
        if features.question > 0.3:
            signals.append("question detected")
        if features.request_language > 0.2:
            signals.append("request language")
        if features.technical_terms > 0.2:
            signals.append("technical content")
        if features.recent_interaction > 0.3:
            signals.append("recent interaction")
        if features.background_signal > 0.5:
            signals.append("background speech")
        if not signals:
            signals.append("no strong signals")
        return "; ".join(signals)
