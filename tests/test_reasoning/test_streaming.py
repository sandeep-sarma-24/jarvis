"""Streaming OllamaModel, fallback wrapper and builder. httpx.MockTransport only."""
from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from app.config import ReasoningConfig
from app.reasoning import build_reasoning_model
from app.reasoning.fallback import FallbackModel
from app.reasoning.local_model import OllamaModel, _ThinkFilter, format_residency
from app.reasoning.model import ReasoningError, StreamingReasoningModel

MSGS = [{"role": "user", "content": "hi"}]
DONE = {"done": True, "eval_count": 20, "eval_duration": 500_000_000, "load_duration": 3_000_000,
        "message": {"content": ""}}


def frame(content: str = "", thinking: str | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if thinking is not None:
        msg["thinking"] = thinking
    return {"done": False, "message": msg}


class Lines(httpx.SyncByteStream):
    """NDJSON body that records whether it was closed and how far it was read."""

    def __init__(self, frames: list, *, raw: bool = False) -> None:
        self.frames = frames
        self.raw = raw
        self.sent = 0
        self.closed = False

    def __iter__(self):
        for f in self.frames:
            self.sent += 1
            yield (f if self.raw else json.dumps(f) + "\n").encode()

    def close(self) -> None:
        self.closed = True


def model_with(body: Lines | None = None, *, status: int = 200, requests: list | None = None,
               handler=None, **kw) -> OllamaModel:
    def default(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(json.loads(request.content))
        return httpx.Response(status, stream=body)
    return OllamaModel(model="m", base_url="http://test", transport=httpx.MockTransport(handler or default), **kw)


class TestStream:
    def test_satisfies_protocol(self):
        assert isinstance(model_with(Lines([])), StreamingReasoningModel)

    def test_deltas_in_order_and_stats(self):
        m = model_with(Lines([frame("Hello"), frame(" there"), frame("."), DONE]))
        assert list(m.stream(MSGS)) == ["Hello", " there", "."]
        s = m.last_stats
        assert s["output_tokens"] == 20
        assert s["tokens_per_second"] == 40.0  # 20 tokens / 0.5 s eval_duration
        assert s["load_ms"] == 3.0
        assert s["ttft_ms"] >= 0

    def test_payload(self):
        reqs: list = []
        m = model_with(Lines([DONE]), requests=reqs, keep_alive="10m", think=False)
        list(m.stream(MSGS, max_tokens=64, temperature=0.5))
        p = reqs[0]
        assert p["stream"] is True and p["keep_alive"] == "10m" and p["think"] is False
        assert p["options"] == {"num_predict": 64, "temperature": 0.5}
        assert p["model"] == "m" and p["messages"] == MSGS

    def test_thinking_field_never_yielded(self):
        m = model_with(Lines([frame("", thinking="secret plan"), frame("Answer", thinking="more"), DONE]))
        assert "".join(m.stream(MSGS)) == "Answer"

    def test_inline_think_block_split_across_deltas(self):
        parts = ["<thi", "nk>I sho", "uld say hi</th", "ink>\n\nHi", " <", "b>there</b>"]
        m = model_with(Lines([frame(p) for p in parts] + [DONE]))
        assert "".join(m.stream(MSGS)) == "Hi <b>there</b>"

    def test_cancel_aborts_and_closes_response(self):
        cancel = threading.Event()
        body = Lines([frame("a"), frame("b"), frame("c"), DONE])
        m = model_with(body)
        out = []
        for d in m.stream(MSGS, cancel=cancel):
            out.append(d)
            cancel.set()
        assert out == ["a"]
        assert body.closed and body.sent == 2

    def test_generator_close_closes_response(self):
        body = Lines([frame("a"), frame("b"), DONE])
        gen = model_with(body).stream(MSGS)
        assert next(gen) == "a"
        gen.close()
        assert body.closed

    def test_connect_failure_raises_quickly(self):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)
        m = model_with(handler=refuse)
        t0 = time.perf_counter()
        with pytest.raises(ReasoningError, match="ConnectError"):
            list(m.stream(MSGS))
        assert time.perf_counter() - t0 < 1

    def test_http_error_raises(self):
        with pytest.raises(ReasoningError, match="HTTP 404"):
            list(model_with(Lines([{"error": "model not found"}]), status=404).stream(MSGS))

    def test_error_frame_raises(self):
        with pytest.raises(ReasoningError, match="boom"):
            list(model_with(Lines([frame("a"), {"error": "boom"}])).stream(MSGS))

    def test_truncated_stream_raises(self):
        with pytest.raises(ReasoningError, match="without done"):
            list(model_with(Lines([frame("a"), frame("b")])).stream(MSGS))

    def test_malformed_line_raises(self):
        with pytest.raises(ReasoningError, match="malformed"):
            list(model_with(Lines(['{"done": false, "mess\n'], raw=True)).stream(MSGS))

    def test_timeouts_split(self):
        m = OllamaModel(timeout_seconds=30, connect_timeout_seconds=2)
        assert m._client.timeout.connect == 2 and m._client.timeout.read == 30
        m.close()

    def test_cancel_unblocks_a_stalled_read(self):
        """No line is coming (model loading); cancel must shut the socket rather than wait."""
        shut = threading.Event()

        class Sock:
            def shutdown(self, how):
                shut.set()

        class NetStream:
            def get_extra_info(self, info):
                return Sock() if info == "socket" else None

        class Stalled(httpx.SyncByteStream):
            def __iter__(self):
                assert shut.wait(5), "socket was never shut down"
                raise httpx.ReadError("connection shut")

        def handler(request):
            return httpx.Response(200, stream=Stalled(), extensions={"network_stream": NetStream()})
        m = model_with(handler=handler)
        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        t0 = time.perf_counter()
        assert list(m.stream(MSGS, cancel=cancel)) == []  # returns quietly, no ReasoningError
        assert time.perf_counter() - t0 < 1
        assert not m._socks

    def test_close_during_stream_from_other_thread(self):
        body = Lines([frame("a"), frame("b"), DONE])
        m = model_with(body)
        gen = m.stream(MSGS)
        assert next(gen) == "a"
        t = threading.Thread(target=m.close)
        t.start()
        t.join()
        assert list(gen) == ["b"]  # MockTransport has no socket to tear down; must not crash


class TestThinkFilter:
    @pytest.mark.parametrize("text", ["plain", "a < b", "ends with <", "<thinking is fine>"])
    def test_passthrough_char_by_char(self, text):
        f = _ThinkFilter()
        assert "".join(f.feed(c) for c in text) + f.flush() == text

    def test_unclosed_block_dropped(self):
        f = _ThinkFilter()
        assert f.feed("ok <think>never closed") + f.flush() == "ok "


class TestGenerateAndWarmUp:
    def test_generate_payload_and_result(self):
        reqs: list = []

        def handler(request):
            reqs.append(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": "<think>x</think> 4 "},
                                             "eval_count": 3, "prompt_eval_count": 7})
        r = model_with(handler=handler).generate(MSGS, max_tokens=10)
        assert r.success and r.text == "4" and r.output_tokens == 3 and r.input_tokens == 7
        assert reqs[0]["stream"] is False and reqs[0]["keep_alive"] == "30m" and reqs[0]["think"] is False

    def test_generate_failure_is_result(self):
        r = model_with(Lines([]), status=500).generate(MSGS)
        assert not r.success and "500" in r.error

    def test_warm_up(self):
        reqs: list = []

        def handler(request):
            reqs.append(json.loads(request.content))
            return httpx.Response(200, json={"done": True})
        assert model_with(handler=handler).warm_up() is True
        assert reqs[0]["messages"] == [] and reqs[0]["keep_alive"] == "30m"

    def test_warm_up_never_raises(self):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)
        assert model_with(handler=refuse).warm_up() is False


class FakeModel:
    def __init__(self, name: str, deltas=("ok",), fail_before: bool = False, fail_after: bool = False):
        self.model_name = name
        self.deltas = list(deltas)
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.calls = 0
        self.closed = False
        self.last_stats = {"ttft_ms": 1.0}
        self.warm = 0

    def stream(self, messages, *, max_tokens=256, temperature=0.2, cancel=None):
        self.calls += 1
        if self.fail_before:
            raise ReasoningError(f"{self.model_name} down")
        yield from self.deltas
        if self.fail_after:
            raise ReasoningError("mid-stream")

    def generate(self, messages, *, max_tokens=256, temperature=0.2):
        from app.reasoning.model import ReasoningResult
        self.calls += 1
        ok = not self.fail_before
        return ReasoningResult(text="".join(self.deltas) if ok else "", model=self.model_name,
                               latency_ms=1.0, success=ok, error=None if ok else "ConnectError")

    def warm_up(self):
        self.warm += 1
        return not self.fail_before

    def close(self):
        self.closed = True


class TestFallback:
    def test_primary_used_when_healthy(self):
        p, s = FakeModel("p", ["a", "b"]), FakeModel("s")
        fb = FallbackModel(p, s)
        assert list(fb.stream(MSGS)) == ["a", "b"] and s.calls == 0
        assert fb.model_name == "p"

    def test_fallback_before_first_token_and_cooldown(self, monkeypatch, caplog):
        now = [100.0]
        monkeypatch.setattr("app.reasoning.fallback.time.monotonic", lambda: now[0])
        p, s = FakeModel("p", fail_before=True), FakeModel("s", ["local"])
        fb = FallbackModel(p, s, cooldown_seconds=30)
        assert list(fb.stream(MSGS)) == ["local"]
        assert fb.model_name == "s" and fb.last_stats is s.last_stats
        now[0] += 10
        assert list(fb.stream(MSGS)) == ["local"]
        assert p.calls == 1  # skipped during cooldown
        assert sum("switching to fallback" in r.message for r in caplog.records) == 1
        now[0] += 25
        p.fail_before = False
        assert list(fb.stream(MSGS)) == ["ok"]
        assert p.calls == 2 and fb.model_name == "p"

    def test_no_fallback_after_first_delta(self):
        p, s = FakeModel("p", ["a"], fail_after=True), FakeModel("s")
        out = []
        with pytest.raises(ReasoningError, match="mid-stream"):
            for d in FallbackModel(p, s).stream(MSGS):
                out.append(d)
        assert out == ["a"] and s.calls == 0

    def test_closing_wrapper_closes_primary_stream(self):
        body = Lines([frame("a"), frame("b"), DONE])
        fb = FallbackModel(model_with(body), FakeModel("s"))
        gen = fb.stream(MSGS)
        assert next(gen) == "a"
        gen.close()
        assert body.closed

    def test_generate_fallback(self):
        p, s = FakeModel("p", fail_before=True), FakeModel("s", ["local"])
        fb = FallbackModel(p, s)
        r = fb.generate(MSGS)
        assert r.success and r.text == "local"
        fb.generate(MSGS)
        assert p.calls == 1

    def test_warm_up_only_loads_secondary_when_primary_down(self):
        p, s = FakeModel("p"), FakeModel("s")
        FallbackModel(p, s).warm_up()
        assert (p.warm, s.warm) == (1, 0)
        p.fail_before = True
        fb = FallbackModel(p, s)
        fb.warm_up()
        assert s.warm == 1
        list(fb.stream(MSGS))
        assert p.calls == 0  # primary in cooldown after failed warm-up

    def test_close_closes_both(self):
        p, s = FakeModel("p"), FakeModel("s")
        FallbackModel(p, s).close()
        assert p.closed and s.closed


class TestBuild:
    def test_without_fallback(self):
        m = build_reasoning_model(ReasoningConfig(model="x", fallback_model=""))
        assert isinstance(m, OllamaModel) and m.model_name == "x"
        m.close()

    def test_with_fallback(self):
        cfg = ReasoningConfig(model="x", ollama_url="http://a:1", fallback_model="y",
                              fallback_url="http://b:2", fallback_cooldown_seconds=5,
                              connect_timeout_seconds=1.5, timeout_seconds=20)
        m = build_reasoning_model(cfg)
        assert isinstance(m, FallbackModel) and isinstance(m, StreamingReasoningModel)
        assert m._primary._base_url == "http://a:1" and m._secondary.model_name == "y"
        assert m._secondary._base_url == "http://b:2" and m._cooldown == 5
        assert m._secondary._client.timeout.connect == 1.5 and m._secondary._client.timeout.read == 20
        m.close()


class TestLocalRuntime:
    def test_num_ctx_in_chat_and_warm_up_payloads(self):
        reqs: list = []

        def handler(request):
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": []})
            reqs.append(json.loads(request.content))
            if reqs[-1]["stream"]:
                return httpx.Response(200, stream=Lines([DONE]))
            return httpx.Response(200, json={"done": True})
        m = model_with(handler=handler, num_ctx=4096)
        assert m.warm_up() is True
        list(m.stream(MSGS, max_tokens=8))
        assert reqs[0]["options"] == {"num_ctx": 4096}
        assert reqs[1]["options"] == {"num_predict": 8, "temperature": 0.2, "num_ctx": 4096}

    def test_build_passes_num_ctx(self):
        m = build_reasoning_model(ReasoningConfig(model="x", fallback_model="y", num_ctx=8192))
        assert m._primary._num_ctx == 8192 and m._secondary._num_ctx == 8192
        m.close()

    def test_prompt_stats(self):
        done = {**DONE, "prompt_eval_count": 512, "prompt_eval_duration": 250_000_000}
        m = model_with(Lines([frame("a"), done]))
        list(m.stream(MSGS))
        assert m.last_stats["prompt_tokens"] == 512 and m.last_stats["prompt_eval_ms"] == 250.0
        assert m.last_stats["output_tokens"] == 20  # existing keys kept

    def test_connection_refused_names_url_and_fix(self):
        def refuse(request):
            raise httpx.ConnectError("Connection refused", request=request)
        with pytest.raises(ReasoningError) as exc:
            list(model_with(handler=refuse).stream(MSGS))
        msg = str(exc.value)
        assert "http://test" in msg and "ollama serve" in msg and "ollama pull m" in msg

    def test_404_names_pull_command(self):
        with pytest.raises(ReasoningError, match="ollama pull m"):
            list(model_with(Lines([{"error": "model 'm' not found"}]), status=404).stream(MSGS))

    def test_residency(self):
        ps = {"models": [{"name": "other:1", "size": 1}, {"name": "m", "model": "m", "size": 4 * 2**30,
                          "size_vram": 3 * 2**30, "context_length": 4096, "digest": "x"}]}
        m = model_with(handler=lambda r: httpx.Response(200, json=ps))
        r = m.residency()
        assert r == {"name": "m", "size": 4 * 2**30, "size_vram": 3 * 2**30, "context_length": 4096}
        assert format_residency(r) == "m size 4.00 GiB, size_vram 3.00 GiB (75% GPU), context 4096"

    def test_residency_none_when_not_loaded_or_down(self):
        assert model_with(handler=lambda r: httpx.Response(200, json={"models": []})).residency() is None

        def refuse(request):
            raise httpx.ConnectError("refused", request=request)
        assert model_with(handler=refuse).residency() is None
        assert format_residency(None) == "not loaded"

    def test_warm_up_logs_residency(self, caplog):
        def handler(request):
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [{"name": "m", "size": 100, "size_vram": 100}]})
            return httpx.Response(200, json={"done": True})
        with caplog.at_level("INFO", logger="app.reasoning.local_model"):
            assert model_with(handler=handler).warm_up() is True
        assert any("100% GPU" in r.getMessage() for r in caplog.records)
