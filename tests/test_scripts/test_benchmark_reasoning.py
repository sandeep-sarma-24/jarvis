"""benchmark_reasoning / benchmark_mlx_llm / verify_local_gemma helpers. No network, no model."""
import pytest

from scripts import benchmark_mlx_llm, benchmark_reasoning as br, verify_local_gemma


def test_prompt_sizes_are_roughly_as_documented() -> None:
    est = {s: sum(len(m["content"]) for m in br.build_messages(s, "q")) / 4 for s in br.SIZES}
    assert est["short"] < 40 and 450 < est["medium"] < 650 and 2400 < est["long"] < 2700
    msgs = br.build_messages("long", "the question", rotate=1)
    assert msgs[0] == {"role": "system", "content": br.SYSTEM} and msgs[-1]["content"] == "the question"
    assert msgs[1]["content"] == br.HISTORY[1][0]  # rotated past the cached prefix


def test_mlx_copy_matches() -> None:
    for name in ("SYSTEM", "PROMPTS", "HISTORY", "SIZES"):
        assert getattr(benchmark_mlx_llm, name) == getattr(br, name)
    assert benchmark_mlx_llm.build_messages("medium", "q", 2) == br.build_messages("medium", "q", 2)


@pytest.mark.parametrize("mod", [br, benchmark_mlx_llm])
def test_memory_parsers(mod) -> None:
    assert mod.parse_memory_free_percent("...\nSystem-wide memory free percentage: 63%\n") == 63.0
    assert mod.parse_memory_free_percent("") is None
    assert mod.parse_swap_used_mb("total = 2048.00M  used = 512.25M  free = 1535.75M  (encrypted)") == 512.25
    assert mod.parse_swap_used_mb("total = 0.00M  used = 0.00M  free = 0.00M") == 0.0
    assert mod.parse_swap_used_mb("") is None


def test_is_pulled() -> None:
    tags = {"models": [{"name": "gemma4:e2b"}, {"name": "llama:latest"}]}
    assert br.is_pulled(tags, "gemma4:e2b") and br.is_pulled(tags, "llama")
    assert not br.is_pulled(tags, "gemma4:12b")


def test_summarize_skips_missing_metrics() -> None:
    s = br.summarize([{"ttft_ms": 10.0, "total_ms": 20.0}, {"total_ms": 30.0}])
    assert s["ttft_ms"]["n"] == 1 and s["total_ms"]["p50"] == 20.0 and s["prompt_tokens"] == {"n": 0}


def test_ollama_rss_is_a_number() -> None:
    assert br.ollama_rss_mb() >= 0


def test_unreachable_target_prints_not_run(monkeypatch, capsys) -> None:
    import httpx

    def refuse(*a, **kw):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(br.httpx, "get", refuse)
    out = br._benchmark_target("http://127.0.0.1:9", "m", [4096], 1, 8)
    assert "not_run" in out and "NOT RUN" in capsys.readouterr().out


def test_verify_check_reply() -> None:
    assert verify_local_gemma.check_reply("local ok.", 3) is None
    assert "more than one chunk" in verify_local_gemma.check_reply("LOCAL OK", 1)
    assert "does not contain" in verify_local_gemma.check_reply("hello there", 4)


def test_verify_refuses_remote_url(monkeypatch, capsys) -> None:
    from app.config import JarvisConfig
    cfg = JarvisConfig()
    cfg.reasoning.ollama_url = "http://gpu-host:11434"
    monkeypatch.setattr(verify_local_gemma, "load_config", lambda: cfg)
    assert verify_local_gemma.run() == 1
    assert "not on this machine" in capsys.readouterr().out


def test_verify_model_not_pulled(monkeypatch, capsys) -> None:
    import httpx
    monkeypatch.setattr(verify_local_gemma.httpx, "get",
                        lambda *a, **kw: httpx.Response(200, json={"models": []}, request=httpx.Request("GET", "http://x")))
    assert verify_local_gemma.run() == 1
    assert "ollama pull" in capsys.readouterr().out
