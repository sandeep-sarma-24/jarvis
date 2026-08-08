"""Tests for ResponseDecision."""
import pytest

from app.reasoning.model import ReasoningResult
from app.response.decision import ResponseDecision, decide


class TestDecidePolicy:

    def test_successful_reasoning_responds(self):
        result = ReasoningResult(
            text="The answer is 42.", model="test", latency_ms=100.0, success=True,
        )
        resp = decide("e1", result)
        assert resp.respond is True
        assert resp.text == "The answer is 42."
        assert resp.event_id == "e1"

    def test_failed_reasoning_no_response(self):
        result = ReasoningResult(
            text="", model="test", latency_ms=50.0,
            success=False, error="timeout",
        )
        resp = decide("e2", result)
        assert resp.respond is False
        assert resp.text is None
        assert "failed" in resp.reason

    def test_empty_text_no_response(self):
        result = ReasoningResult(
            text="", model="test", latency_ms=50.0, success=True,
        )
        resp = decide("e3", result)
        assert resp.respond is False
        assert "empty" in resp.reason

    def test_whitespace_text_no_response(self):
        result = ReasoningResult(
            text="   \n  ", model="test", latency_ms=50.0, success=True,
        )
        resp = decide("e4", result)
        assert resp.respond is False

    def test_event_id_preserved(self):
        result = ReasoningResult(
            text="hi", model="test", latency_ms=1.0, success=True,
        )
        resp = decide("my-event-id", result)
        assert resp.event_id == "my-event-id"


class TestResponseDecisionShape:

    def test_frozen(self):
        resp = ResponseDecision(
            respond=True, text="hi", reason="test", event_id="e1",
        )
        with pytest.raises(AttributeError):
            resp.respond = False

    def test_fields(self):
        resp = ResponseDecision(
            respond=True, text="hello", reason="test reason", event_id="eid",
        )
        assert isinstance(resp.respond, bool)
        assert isinstance(resp.text, str)
        assert isinstance(resp.reason, str)
        assert isinstance(resp.event_id, str)


class TestSafeResponses:

    def test_failure_never_hallucinates(self):
        """A failed result must never produce a response with hallucinated text."""
        result = ReasoningResult(
            text="This should not appear", model="test",
            latency_ms=1.0, success=False, error="network error",
        )
        resp = decide("e1", result)
        assert resp.respond is False
        assert resp.text is None
