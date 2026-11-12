"""Incremental sentence splitter for streaming LLM output → TTS.

A sentence ends at . ! ? (plus closing quotes/brackets) followed by
whitespace, or at a newline. Requiring the following whitespace is what keeps
"3.5" and an unfinished "3." in one piece until more text arrives.
"""
from __future__ import annotations

import re

_END = re.compile(r"[.!?]+[\"'”’)\]]*\s|\n")
_CLAUSE = re.compile(r"[,;:]\s")

_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "approx", "no", "fig", "inc", "ltd", "co", "mt", "jan", "feb", "aug", "sept",
    "oct", "nov", "dec", "a.m", "p.m", "u.s",
})


class SentenceSplitter:
    """feed() returns sentences as soon as they are complete; flush() the rest.

    min_chars: fragments shorter than this ("Sure.", "1.") are merged with the
      following sentence instead of being spoken alone.
    first_clause_chars: until the first sentence is out, a long opening
      sentence is cut at a , ; or : once it is at least this long, so first
      audio is not held back by a long opening sentence.
    """

    def __init__(self, *, min_chars: int = 10, first_clause_chars: int = 40) -> None:
        self._min = min_chars
        self._first_clause = first_clause_chars
        self._buf = ""
        self._emitted = False

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            sentence = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
                self._emitted = True
        return out

    def flush(self) -> str | None:
        rest, self._buf = self._buf.strip(), ""
        return rest or None

    def _find_cut(self) -> int | None:
        for m in _END.finditer(self._buf):
            head = self._buf[: m.start()]
            if m.group() != "\n" and m.group().startswith(".") and not m.group().startswith(".."):
                token = head.rsplit(None, 1)[-1].lstrip("(\"'") if head.strip() else ""
                if token.lower() in _ABBREVIATIONS or (len(token) == 1 and token.isalpha() and token != "I"):
                    continue
            if len(head.strip()) + 1 < self._min:
                continue
            return m.end()
        if not self._emitted:
            for m in _CLAUSE.finditer(self._buf):
                if m.start() >= self._first_clause:
                    return m.end()
        return None
