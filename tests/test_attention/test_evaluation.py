"""Evaluation harness — runs the curated attention dataset against the engine."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from app.attention.engine import AttentionEngine
from app.events.types import Modality, SpeechEvent

_DATASET = Path(__file__).resolve().parent.parent / "data" / "attention_cases.json"


def _load_cases() -> list[dict]:
    with open(_DATASET) as f:
        return json.load(f)


def _speech(text: str) -> SpeechEvent:
    return SpeechEvent(session_id="eval", source=Modality.AUDIO,
                       transcript=text, duration_seconds=1.0)


def test_evaluation_accuracy():
    """Run curated dataset and report accuracy. Fails if overall <80%."""
    cases = _load_cases()
    engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)

    total = 0
    correct = 0
    incorrect_cases = []
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for case in cases:
        text = case["text"]
        expected = case["expected_should_reason"]
        category = case["category"]

        decision = engine.evaluate(_speech(text))
        total += 1
        by_category[category]["total"] += 1

        if decision.should_reason == expected:
            correct += 1
            by_category[category]["correct"] += 1
        else:
            incorrect_cases.append({
                "text": text,
                "expected": expected,
                "got": decision.should_reason,
                "salience": decision.salience,
                "tier": decision.tier,
                "reason": decision.reason,
                "category": category,
            })

    accuracy = correct / total if total else 0

    # Report
    report = [
        f"\n{'='*60}",
        f"ATTENTION EVALUATION REPORT",
        f"{'='*60}",
        f"Total:    {total}",
        f"Correct:  {correct}",
        f"Incorrect:{total - correct}",
        f"Accuracy: {accuracy:.1%}",
        f"",
        f"reason_true:  {engine.counters['reason_true']}",
        f"reason_false: {engine.counters['reason_false']}",
        f"",
        "Category breakdown:",
    ]
    for cat, counts in sorted(by_category.items()):
        cat_acc = counts["correct"] / counts["total"] if counts["total"] else 0
        report.append(f"  {cat:20s}  {counts['correct']}/{counts['total']}  ({cat_acc:.0%})")

    if incorrect_cases:
        report.append("")
        report.append("Incorrect cases:")
        for ic in incorrect_cases:
            report.append(f"  [{ic['category']}] {ic['text']!r}")
            report.append(f"    expected={ic['expected']} got={ic['got']} "
                          f"salience={ic['salience']:.2f} tier={ic['tier']}")

    print("\n".join(report))

    assert accuracy >= 0.80, f"Accuracy {accuracy:.1%} below 80% threshold"
