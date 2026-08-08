"""Ollama-backed local reasoning model.

Talks to the Ollama REST API (OpenAI-compatible chat endpoint).
Synchronous — designed to be called via asyncio.to_thread so it
doesn't block the event loop.
"""
from __future__ import annotations

import time

import httpx

from app.reasoning.model import ReasoningModel, ReasoningResult


class OllamaModel:
    """Reasoning model backed by a local Ollama instance.

    ponytail: uses httpx synchronously. The caller wraps in asyncio.to_thread.
    Upgrade path: switch to async httpx client if profiling shows thread overhead.
    """

    def __init__(
        self,
        *,
        model: str = "gemma4:12b",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> ReasoningResult:
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            text = data.get("message", {}).get("content", "")
            latency = (time.perf_counter() - t0) * 1000

            eval_count = data.get("eval_count")
            prompt_eval_count = data.get("prompt_eval_count")

            return ReasoningResult(
                text=text.strip(),
                model=self._model,
                latency_ms=round(latency, 1),
                success=True,
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return ReasoningResult(
                text="",
                model=self._model,
                latency_ms=round(latency, 1),
                success=False,
                error=str(exc),
            )

    def close(self) -> None:
        self._client.close()
