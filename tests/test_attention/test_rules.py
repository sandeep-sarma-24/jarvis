"""Tests for Tier 1 deterministic attention rules."""
import pytest

from app.attention.rules import AttentionRules, detect_question, detect_request


@pytest.fixture
def rules():
    return AttentionRules(assistant_names=["jarvis"])


# ── Explicit address ────────────────────────────────────────────────────

class TestExplicitAddress:

    def test_hey_jarvis_question(self, rules):
        r = rules.evaluate("Hey Jarvis, what time is it?")
        assert r is not None
        assert r.should_reason is True
        assert r.salience >= 0.9

    def test_jarvis_at_start(self, rules):
        r = rules.evaluate("Jarvis, explain this API error.")
        assert r is not None
        assert r.should_reason is True
        assert r.salience >= 0.9

    def test_hey_jarvis_alone(self, rules):
        r = rules.evaluate("Hey JARVIS.")
        assert r is not None
        assert r.should_reason is True

    def test_jarvis_at_end(self, rules):
        r = rules.evaluate("Can you help me, Jarvis?")
        assert r is not None
        assert r.should_reason is True

    def test_case_insensitive(self, rules):
        r = rules.evaluate("HEY JARVIS!!! WHAT IS THIS?")
        assert r is not None
        assert r.should_reason is True
        assert r.salience >= 0.9

    def test_ok_jarvis(self, rules):
        r = rules.evaluate("Ok Jarvis, run the tests")
        assert r is not None
        assert r.should_reason is True


class TestFalseAddress:

    def test_jarvis_fictional_character(self, rules):
        r = rules.evaluate("Jarvis is a fictional character.")
        assert r is not None
        assert r.should_reason is False

    def test_jarvis_iron_man(self, rules):
        r = rules.evaluate("I was reading about Jarvis in Iron Man.")
        # Should not trigger high-confidence attention
        assert r is None or r.should_reason is False

    def test_jarvis_the_butler(self, rules):
        r = rules.evaluate("Jarvis was the butler in the comics.")
        assert r is not None
        assert r.should_reason is False


# ── Background speech ───────────────────────────────────────────────────

class TestBackgroundSpeech:

    def test_call_tomorrow(self, rules):
        r = rules.evaluate("I'll call you tomorrow.")
        assert r is not None
        assert r.should_reason is False

    def test_yeah_fine(self, rules):
        r = rules.evaluate("Yeah, that's fine.")
        assert r is not None
        assert r.should_reason is False

    def test_lets_go(self, rules):
        r = rules.evaluate("Let's go.")
        assert r is not None
        assert r.should_reason is False

    def test_goodbye(self, rules):
        r = rules.evaluate("Goodbye!")
        assert r is not None
        assert r.should_reason is False


# ── Empty / short ───────────────────────────────────────────────────────

class TestEmpty:

    def test_empty_string(self, rules):
        r = rules.evaluate("")
        assert r is not None
        assert r.should_reason is False
        assert r.salience == 0.0

    def test_whitespace(self, rules):
        r = rules.evaluate("   ")
        assert r is not None
        assert r.should_reason is False

    def test_single_char(self, rules):
        r = rules.evaluate("a")
        assert r is not None
        assert r.should_reason is False


# ── Ambiguous (should defer to scorer) ──────────────────────────────────

class TestDeferred:

    def test_question_deferred(self, rules):
        r = rules.evaluate("Why is this API failing?")
        assert r is None

    def test_statement_deferred(self, rules):
        r = rules.evaluate("That's interesting.")
        assert r is None

    def test_technical_deferred(self, rules):
        r = rules.evaluate("The server is throwing a 500 error.")
        assert r is None


# ── Custom assistant names ──────────────────────────────────────────────

class TestCustomNames:

    def test_custom_name(self):
        rules = AttentionRules(assistant_names=["friday"])
        r = rules.evaluate("Hey Friday, what's up?")
        assert r is not None
        assert r.should_reason is True

    def test_multiple_names(self):
        rules = AttentionRules(assistant_names=["jarvis", "friday"])
        r1 = rules.evaluate("Hey Jarvis, help me.")
        r2 = rules.evaluate("Friday, what time is it?")
        assert r1 is not None and r1.should_reason is True
        assert r2 is not None and r2.should_reason is True


# ── Question detection ──────────────────────────────────────────────────

class TestDetectQuestion:

    def test_question_mark(self):
        assert detect_question("What time is it?") > 0.4

    def test_question_word_start(self):
        assert detect_question("Why is this failing") > 0.3

    def test_how_does(self):
        assert detect_question("How does this work?") > 0.5

    def test_aux_start(self):
        assert detect_question("Can you explain this?") > 0.4

    def test_statement(self):
        assert detect_question("I'll call you tomorrow.") == 0.0

    def test_single_word_question(self):
        score = detect_question("What?")
        assert score > 0.0
        longer_score = detect_question("Why is the API returning 500?")
        assert longer_score >= score

    def test_empty(self):
        assert detect_question("") == 0.0


# ── Request detection ───────────────────────────────────────────────────

class TestDetectRequest:

    def test_explain(self):
        assert detect_request("Explain this error.") > 0.3

    def test_can_you(self):
        assert detect_request("Can you help me with this?") > 0.2

    def test_tell_me(self):
        assert detect_request("Tell me about the deployment process.") > 0.3

    def test_no_request(self):
        assert detect_request("The weather is nice today.") == 0.0

    def test_empty(self):
        assert detect_request("") == 0.0
