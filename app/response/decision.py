"""Response Decision — separates reasoning from responding.

Attention answers: "Should I reason?"
Response Decision answers: "Given reasoning, should I respond?"

Phase 3 policy is deterministic: if attention was positive and reasoning
succeeded, respond with the reasoning text. No second LLM call.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.reasoning.model import ReasoningResult

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass(frozen=True)
class ResponseDecision:
    respond: bool
    text: str | None
    thinking: str | None
    reason: str
    event_id: str


def _split_thinking(raw: str) -> tuple[str | None, str]:
    """Extract <think> block from model output. Returns (thinking, visible_text)."""
    match = _THINK_RE.search(raw)
    if not match:
        return None, raw.strip()
    thinking = match.group(1).strip()
    visible = _THINK_RE.sub("", raw).strip()
    return (thinking or None), visible


def decide(
    event_id: str,
    reasoning_result: ReasoningResult,
) -> ResponseDecision:
    """Deterministic response policy for Phase 3.

    - Reasoning succeeded with text → respond.
    - Reasoning failed → no response.
    - Reasoning produced empty text → no response.

    Thinking models (e.g. Gemma 4) emit <think>...</think> blocks.
    These are extracted and stored separately — never sent to TTS.
    """
    if not reasoning_result.success:
        return ResponseDecision(
            respond=False,
            text=None,
            thinking=None,
            reason=f"reasoning failed: {reasoning_result.error}",
            event_id=event_id,
        )

    thinking, visible = _split_thinking(reasoning_result.text)

    if not visible:
        logger.debug("Raw reasoning text (empty visible): %r", reasoning_result.text[:500])
        return ResponseDecision(
            respond=False,
            text=None,
            thinking=thinking,
            reason="reasoning produced empty response",
            event_id=event_id,
        )

    return ResponseDecision(
        respond=True,
        text=visible,
        thinking=thinking,
        reason="attention-positive request with successful reasoning",
        event_id=event_id,
    )
