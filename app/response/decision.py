"""Response Decision — separates reasoning from responding.

Attention answers: "Should I reason?"
Response Decision answers: "Given reasoning, should I respond?"

Phase 3 policy is deterministic: if attention was positive and reasoning
succeeded, respond with the reasoning text. No second LLM call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.reasoning.model import ReasoningResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResponseDecision:
    respond: bool
    text: str | None
    reason: str
    event_id: str


def decide(
    event_id: str,
    reasoning_result: ReasoningResult,
) -> ResponseDecision:
    """Deterministic response policy for Phase 3.

    - Reasoning succeeded with text → respond.
    - Reasoning failed → no response.
    - Reasoning produced empty text → no response.

    ponytail: Phase 5+ may add structured output parsing or a second
    LLM pass for silent-reasoning cases. For now, attention-positive +
    successful reasoning = respond.
    """
    if not reasoning_result.success:
        return ResponseDecision(
            respond=False,
            text=None,
            reason=f"reasoning failed: {reasoning_result.error}",
            event_id=event_id,
        )

    text = reasoning_result.text.strip()
    if not text:
        return ResponseDecision(
            respond=False,
            text=None,
            reason="reasoning produced empty response",
            event_id=event_id,
        )

    return ResponseDecision(
        respond=True,
        text=text,
        reason="attention-positive request with successful reasoning",
        event_id=event_id,
    )
