"""Tests for ReasoningModel protocol and ReasoningResult."""
import pytest

from app.reasoning.model import ReasoningModel, ReasoningResult


class FakeModel:
    """Deterministic fake model for testing."""

    def __init__(self, response: str = "This is a test response."):
        self._response = response

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ReasoningResult:
        return ReasoningResult(
            text=self._response,
            model=self.model_name,
            latency_ms=1.0,
            success=True,
        )


class FailingModel:
    """Model that always fails."""

    @property
    def model_name(self) -> str:
        return "failing-model"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ReasoningResult:
        return ReasoningResult(
            text="",
            model=self.model_name,
            latency_ms=0.5,
            success=False,
            error="connection refused",
        )


class TestProtocolCompliance:

    def test_fake_model_satisfies_protocol(self):
        model = FakeModel()
        assert isinstance(model, ReasoningModel)

    def test_failing_model_satisfies_protocol(self):
        model = FailingModel()
        assert isinstance(model, ReasoningModel)


class TestFakeModel:

    def test_generates_response(self):
        model = FakeModel("hello world")
        result = model.generate([{"role": "user", "content": "hi"}])
        assert result.text == "hello world"
        assert result.success is True
        assert result.model == "fake-model"

    def test_custom_response(self):
        model = FakeModel("custom answer")
        result = model.generate([{"role": "user", "content": "test"}])
        assert result.text == "custom answer"


class TestFailingModel:

    def test_returns_failure(self):
        model = FailingModel()
        result = model.generate([{"role": "user", "content": "test"}])
        assert result.success is False
        assert result.error is not None
        assert result.text == ""


class TestReasoningResult:

    def test_frozen(self):
        r = ReasoningResult(text="x", model="m", latency_ms=1.0, success=True)
        with pytest.raises(AttributeError):
            r.text = "y"

    def test_optional_fields(self):
        r = ReasoningResult(text="x", model="m", latency_ms=1.0, success=True)
        assert r.error is None
        assert r.input_tokens is None
        assert r.output_tokens is None

    def test_with_tokens(self):
        r = ReasoningResult(
            text="x", model="m", latency_ms=1.0, success=True,
            input_tokens=100, output_tokens=50,
        )
        assert r.input_tokens == 100
        assert r.output_tokens == 50
