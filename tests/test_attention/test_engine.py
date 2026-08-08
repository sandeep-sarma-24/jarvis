"""Tests for the AttentionEngine orchestrator."""
import pytest

from app.attention.engine import AttentionDecision, AttentionEngine
from app.events.types import Modality, SpeechEvent
from app.memory.working import WorkingMemory


def _speech(transcript: str, **kwargs) -> SpeechEvent:
    return SpeechEvent(
        session_id="test",
        source=Modality.AUDIO,
        transcript=transcript,
        duration_seconds=1.0,
        **kwargs,
    )


@pytest.fixture
def engine():
    return AttentionEngine(
        assistant_names=["jarvis"],
        reason_threshold=0.4,
    )


# ── Core decisions ──────────────────────────────────────────────────────

class TestCoreDecisions:

    def test_explicit_address_reasons(self, engine):
        d = engine.evaluate(_speech("Hey Jarvis, what time is it?"))
        assert d.should_reason is True
        assert d.salience >= 0.9
        assert d.tier == "rule"

    def test_jarvis_explain(self, engine):
        d = engine.evaluate(_speech("Jarvis, explain this API error."))
        assert d.should_reason is True

    def test_background_does_not_reason(self, engine):
        d = engine.evaluate(_speech("I'll call you tomorrow."))
        assert d.should_reason is False

    def test_ambiguous_does_not_reason(self, engine):
        d = engine.evaluate(_speech("That's interesting."))
        assert d.should_reason is False

    def test_empty_does_not_reason(self, engine):
        d = engine.evaluate(_speech(""))
        assert d.should_reason is False
        assert d.salience == 0.0

    def test_question_may_reason(self, engine):
        d = engine.evaluate(_speech("Why is the API returning 500?"))
        # Question with technical content should have elevated salience
        assert d.salience > 0.2


class TestDirectAddressAlwaysWins:
    """Explicit addressing must be resolved by Tier 1 rules with high salience."""

    def test_jarvis_can_you_is_rule_tier(self, engine):
        d = engine.evaluate(_speech("Jarvis, can you explain this API error?"))
        assert d.should_reason is True
        assert d.tier == "rule"
        assert d.salience >= 0.9

    def test_hey_jarvis_is_rule_tier(self, engine):
        d = engine.evaluate(_speech("Hey Jarvis!"))
        assert d.should_reason is True
        assert d.tier == "rule"
        assert d.salience >= 0.9

    def test_trailing_jarvis_is_rule_tier(self, engine):
        d = engine.evaluate(_speech("Can you help me, Jarvis?"))
        assert d.should_reason is True
        assert d.tier == "rule"

    def test_jarvis_could_you(self, engine):
        d = engine.evaluate(_speech("Jarvis, could you check the API?"))
        assert d.should_reason is True
        assert d.tier == "rule"

    def test_jarvis_will_you(self, engine):
        d = engine.evaluate(_speech("Jarvis, will you help me?"))
        assert d.should_reason is True
        assert d.tier == "rule"

    def test_jarvis_would_you(self, engine):
        d = engine.evaluate(_speech("Jarvis, would you open this?"))
        assert d.should_reason is True
        assert d.tier == "rule"


class TestFalseAddress:

    def test_jarvis_fictional(self, engine):
        d = engine.evaluate(_speech("Jarvis is a fictional character."))
        assert d.should_reason is False

    def test_all_caps_address_works(self, engine):
        d = engine.evaluate(_speech("HEY JARVIS!!! WHAT IS THIS?"))
        assert d.should_reason is True


# ── Tiers ───────────────────────────────────────────────────────────────

class TestTiers:

    def test_rule_tier(self, engine):
        d = engine.evaluate(_speech("Hey Jarvis"))
        assert d.tier == "rule"

    def test_scorer_tier(self, engine):
        d = engine.evaluate(_speech("The deployment seems stuck."))
        assert d.tier == "scorer"

    def test_empty_is_rule_tier(self, engine):
        d = engine.evaluate(_speech(""))
        assert d.tier == "rule"


# ── Silence-first policy ────────────────────────────────────────────────

class TestBackgroundSpeech:

    def test_background_examples_silent(self, engine):
        for text in [
            "I'll call you tomorrow.",
            "See you later.",
            "Okay, sure.",
            "Let's go.",
        ]:
            d = engine.evaluate(_speech(text))
            assert d.should_reason is False, f"Expected silence for: {text!r}"


class TestAmbiguousSpeech:

    def test_ambiguous_remains_conservative(self, engine):
        for text in [
            "That's interesting.",
            "Maybe.",
            "I don't know.",
            "That's weird.",
            "I see.",
            "Hmm, okay.",
            "Right.",
        ]:
            d = engine.evaluate(_speech(text))
            assert d.should_reason is False, f"Expected silence for: {text!r}"


class TestThresholdBehavior:

    def test_below_threshold_is_false(self):
        engine = AttentionEngine(reason_threshold=0.5)
        # "That's interesting" scores very low — well below any threshold
        d = engine.evaluate(_speech("That's interesting."))
        assert d.should_reason is False

    def test_above_threshold_is_true(self):
        engine = AttentionEngine(reason_threshold=0.1)
        # A question should score above 0.1
        d = engine.evaluate(_speech("What time is it?"))
        assert d.should_reason is True

    def test_at_threshold_is_true(self):
        """score >= threshold → should_reason=True.

        Set threshold to the exact score a known case produces, then verify
        the >= comparison holds."""
        # "What time is it?" scores ~0.35 with default weights.
        # Set threshold to that exact value to test the boundary.
        probe_engine = AttentionEngine(reason_threshold=0.0)
        probe = probe_engine.evaluate(_speech("What time is it?"))
        exact_score = probe.salience

        engine = AttentionEngine(reason_threshold=exact_score)
        d = engine.evaluate(_speech("What time is it?"))
        assert d.salience == exact_score
        assert d.should_reason is True

    def test_threshold_deterministic(self):
        """Same input must produce same decision every time."""
        engine = AttentionEngine(reason_threshold=0.4)
        results = [engine.evaluate(_speech("Why is this failing?")) for _ in range(10)]
        assert all(r.should_reason == results[0].should_reason for r in results)
        assert all(r.salience == results[0].salience for r in results)


# ── Failure handling ────────────────────────────────────────────────────

class TestFailure:

    def test_fallback_on_exception(self):
        engine = AttentionEngine()
        # Monkey-patch rules to throw
        original = engine._rules.evaluate
        def broken_evaluate(text):
            raise ValueError("boom")
        engine._rules.evaluate = broken_evaluate

        d = engine.evaluate(_speech("test"))
        assert d.should_reason is False
        assert d.tier == "fallback"
        assert engine.counters["fallback_decisions"] >= 1

        engine._rules.evaluate = original


# ── Counters ────────────────────────────────────────────────────────────

class TestCounters:

    def test_counters_increment(self, engine):
        engine.evaluate(_speech("Hey Jarvis"))
        engine.evaluate(_speech("whatever"))
        engine.evaluate(_speech(""))

        c = engine.counters
        assert c["evaluations"] == 3
        assert c["reason_true"] >= 1
        assert c["reason_false"] >= 1

    def test_counter_types(self, engine):
        engine.evaluate(_speech("Hey Jarvis, help"))
        engine.evaluate(_speech("The weather is nice."))

        c = engine.counters
        assert c["rule_decisions"] >= 1
        assert c["scorer_decisions"] >= 0


# ── AttentionDecision shape ─────────────────────────────────────────────

class TestDecisionShape:

    def test_decision_fields(self, engine):
        d = engine.evaluate(_speech("Hey Jarvis, test"))
        assert isinstance(d.should_reason, bool)
        assert isinstance(d.salience, float)
        assert isinstance(d.reason, str)
        assert isinstance(d.tier, str)
        assert 0.0 <= d.salience <= 1.0

    def test_decision_is_frozen(self, engine):
        d = engine.evaluate(_speech("test"))
        with pytest.raises(AttributeError):
            d.should_reason = True


# ── Working memory integration ──────────────────────────────────────────

class TestWorkingMemoryIntegration:

    def test_events_recorded(self, engine):
        engine.evaluate(_speech("Hey Jarvis, hello"))
        engine.evaluate(_speech("some background"))
        assert len(engine.working_memory) == 2

    def test_custom_memory(self):
        mem = WorkingMemory(max_size=3)
        engine = AttentionEngine(working_memory=mem)
        for i in range(5):
            engine.evaluate(_speech(f"event {i}"))
        assert len(mem) == 3

    def test_background_speech_does_not_refresh_interaction(self):
        """Background speech (should_reason=False) must NOT update
        last_interaction_time. Only meaningful assistant interactions do."""
        engine = AttentionEngine(assistant_names=["jarvis"], reason_threshold=0.4)
        mem = engine.working_memory

        engine.evaluate(_speech("I'll call John tomorrow."))
        assert mem.last_interaction_time is None

        engine.evaluate(_speech("Yeah, that's fine."))
        assert mem.last_interaction_time is None

        engine.evaluate(_speech("Hey Jarvis, what's the weather?"))
        assert mem.last_interaction_time is not None
        interaction_time = mem.last_interaction_time

        engine.evaluate(_speech("See you later."))
        # Must not have changed
        assert mem.last_interaction_time == interaction_time
