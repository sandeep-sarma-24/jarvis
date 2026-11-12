"""Smoke test for OllamaModel — requires a running Ollama instance.

Marked as 'model' so it's skipped in normal unit test runs.
Run with: pytest -m model tests/test_reasoning/test_local_model.py
"""
import httpx
import pytest

from app.config import load_config
from app.reasoning.local_model import OllamaModel
from app.reasoning.model import ReasoningModel


pytestmark = pytest.mark.model


class TestOllamaModelSmoke:

    @pytest.fixture
    def model(self):
        cfg = load_config().reasoning
        m = OllamaModel(model=cfg.model, base_url=cfg.ollama_url, timeout_seconds=60.0)
        yield m
        m.close()

    def test_satisfies_protocol(self, model):
        assert isinstance(model, ReasoningModel)

    def test_simple_prompt(self, model):
        try:
            httpx.get(f"{model._base_url}/api/tags", timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            pytest.skip(f"Ollama not reachable at {model._base_url}: {exc}")
        result = model.generate(
            [
                {"role": "system", "content": "Answer in one sentence."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        assert result.success, f"Model failed: {result.error}"
        assert len(result.text) > 0
        assert result.latency_ms > 0
        assert result.model == model._model
        print(f"\nOllama response: {result.text}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        if result.output_tokens:
            print(f"Output tokens: {result.output_tokens}")

    def test_model_name(self, model):
        assert model.model_name == load_config().reasoning.model
