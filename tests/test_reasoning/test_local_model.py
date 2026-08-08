"""Smoke test for OllamaModel — requires a running Ollama instance.

Marked as 'model' so it's skipped in normal unit test runs.
Run with: pytest -m model tests/test_reasoning/test_local_model.py
"""
import pytest

from app.reasoning.local_model import OllamaModel
from app.reasoning.model import ReasoningModel


pytestmark = pytest.mark.model


class TestOllamaModelSmoke:

    @pytest.fixture
    def model(self):
        m = OllamaModel(model="gemma4:12b", timeout_seconds=60.0)
        yield m
        m.close()

    def test_satisfies_protocol(self, model):
        assert isinstance(model, ReasoningModel)

    def test_simple_prompt(self, model):
        result = model.generate(
            [
                {"role": "system", "content": "Answer in one sentence."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        assert result.success, f"Model failed: {result.error}"
        assert len(result.text) > 0
        assert result.latency_ms > 0
        assert result.model == "gemma4:12b"
        print(f"\nOllama response: {result.text}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        if result.output_tokens:
            print(f"Output tokens: {result.output_tokens}")

    def test_model_name(self, model):
        assert model.model_name == "gemma4:12b"
