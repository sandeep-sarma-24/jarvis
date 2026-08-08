"""Tier 1 — Deterministic attention rules.

Fast pattern matching that handles obvious cases without scoring.
Each rule returns a RuleResult when decisive, or None to defer to the scorer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleResult:
    should_reason: bool
    salience: float
    reason: str


_QUESTION_WORDS = frozenset(
    ["who", "what", "when", "where", "why", "how", "which", "whose", "whom"]
)
_QUESTION_AUX = frozenset(
    ["is", "are", "was", "were", "do", "does", "did", "can", "could",
     "will", "would", "should", "shall", "may", "might", "has", "have", "had"]
)
_REQUEST_VERBS = frozenset(
    ["explain", "show", "tell", "help", "describe", "find", "list",
     "summarize", "check", "run", "open", "close", "search", "fix", "debug"]
)
_BACKGROUND_PHRASES = [
    re.compile(r"\b(?:i'?ll call you|see you|talk to you|bye|goodbye|see ya)\b", re.I),
    re.compile(r"^(?:yeah|yep|yup|ok|okay|sure|right|uh huh|mm+|hmm+)(?:[,.\s]+[\w']+){0,4}[.!,\s]*$", re.I),
    re.compile(r"^(?:let'?s go|come on|let'?s do (?:it|this))[.!\s]*$", re.I),
]


class AttentionRules:
    """Tier 1 deterministic rules for attention filtering."""

    def __init__(self, assistant_names: list[str] | None = None) -> None:
        self._names = [n.lower() for n in (assistant_names or ["jarvis"])]
        self._build_patterns()

    def _build_patterns(self) -> None:
        escaped = [re.escape(n) for n in self._names]
        name_alt = "|".join(escaped)
        # Explicit address: name at start, optionally preceded by "hey"/"ok"/"yo"
        self._direct_address = re.compile(
            rf"^(?:hey|ok|yo|hi|hello)?\s*(?:{name_alt})\b", re.I
        )
        # Vocative: name followed by comma — "Jarvis, can you..." is direct address
        self._vocative = re.compile(
            rf"^(?:hey|ok|yo|hi|hello)?\s*(?:{name_alt})\s*,", re.I
        )
        # Name at end: "can you help me, jarvis?"
        self._trailing_address = re.compile(
            rf"\b(?:{name_alt})\s*[?.!]*\s*$", re.I
        )
        # Narrative/third-person usage: "jarvis is a character", "about jarvis"
        # Only matches name directly followed by a verb/article (no comma).
        self._false_address = re.compile(
            rf"\b(?:{name_alt})\s+(?:is|was|were|has|had|would|will|can|could"
            rf"|the|a|an|from|in|about|like)\b", re.I
        )

    def evaluate(self, transcript: str) -> RuleResult | None:
        text = transcript.strip()

        if not text:
            return RuleResult(should_reason=False, salience=0.0, reason="empty transcript")

        if len(text) < 2:
            return RuleResult(should_reason=False, salience=0.0, reason="transcript too short")

        # Explicit assistant address
        address = self._check_address(text)
        if address is not None:
            return address

        # Obvious background speech
        bg = self._check_background(text)
        if bg is not None:
            return bg

        return None

    def _check_address(self, text: str) -> RuleResult | None:
        has_vocative = self._vocative.search(text) is not None
        has_direct = self._direct_address.search(text) is not None
        has_trailing = self._trailing_address.search(text) is not None

        if not has_direct and not has_trailing:
            return None

        # Vocative comma ("Jarvis, can you...") is unambiguously direct address.
        # Trailing address ("help me, Jarvis?") likewise.
        # These take priority over false-address patterns.
        if has_vocative or has_trailing:
            return RuleResult(
                should_reason=True, salience=0.95,
                reason="explicitly addressed assistant"
            )

        # Name at start without comma — check for narrative usage.
        # "Jarvis is a character" vs "Jarvis help me" (rare but valid).
        if self._false_address.search(text):
            return RuleResult(
                should_reason=False, salience=0.2,
                reason="assistant name used in narrative context"
            )

        return RuleResult(
            should_reason=True, salience=0.95,
            reason="explicitly addressed assistant"
        )

    def _check_background(self, text: str) -> RuleResult | None:
        for pat in _BACKGROUND_PHRASES:
            if pat.search(text):
                return RuleResult(
                    should_reason=False, salience=0.1,
                    reason="background or social speech detected"
                )
        return None


def detect_question(text: str) -> float:
    """Return a question-likelihood score in [0, 1] based on deterministic signals."""
    text = text.strip()
    if not text:
        return 0.0

    score = 0.0
    words = text.lower().split()

    if text.rstrip().endswith("?"):
        score += 0.50

    if words and words[0] in _QUESTION_WORDS:
        score += 0.35
    elif any(w in _QUESTION_WORDS for w in words[:3]):
        score += 0.15

    if words and words[0] in _QUESTION_AUX:
        score += 0.20

    return min(score, 1.0)


def detect_request(text: str) -> float:
    """Return a request-likelihood score in [0, 1] based on imperative/request patterns."""
    text = text.strip()
    if not text:
        return 0.0

    score = 0.0
    words = text.lower().split()

    if words and words[0] in _REQUEST_VERBS:
        score += 0.40

    request_phrases = [
        r"\b(?:can you|could you|would you|will you|please)\b",
        r"\b(?:help me|tell me|show me|let me know)\b",
        r"\b(?:i need|i want)\b",
    ]
    for pat in request_phrases:
        if re.search(pat, text, re.I):
            score += 0.25

    return min(score, 1.0)
