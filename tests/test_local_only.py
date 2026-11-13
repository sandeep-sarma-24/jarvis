"""The whole runtime stays on this machine: no remote model host, no cloud LLM API, no silent fallback."""
from __future__ import annotations

import dataclasses
import ipaddress
import re
import socket
import tomllib
from pathlib import Path

import pytest

from app.config import ConfigError, JarvisConfig, enforce_local_only, is_loopback, load_config

ROOT = Path(__file__).resolve().parents[1]
TOML_PATH = ROOT / "configs" / "default.toml"


def _strings(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)
    elif isinstance(node, str):
        yield node


def _source_files():
    for d in ("app", "configs"):
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and p.suffix in {".py", ".toml", ".txt"}:
                yield p


# (a)
def test_shipped_config_has_only_loopback_http_hosts():
    with open(TOML_PATH, "rb") as f:
        raw = tomllib.load(f)
    urls = [s for s in _strings(raw) if "://" in s and s.split("://")[0] in {"http", "https"}]
    assert urls, "expected model URLs in the shipped config"
    assert all(is_loopback(u) for u in urls), urls


# (b)
_TAILSCALE = ipaddress.ip_network("100.64.0.0/10")
_IPV4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


def test_no_tailscale_address_or_remote_host_name_in_source():
    for p in _source_files():
        text = p.read_text()
        assert not re.search(r"tailscale|\bomen\b", text, re.IGNORECASE), p
        for ip in _IPV4.findall(text):
            try:
                assert ipaddress.ip_address(ip) not in _TAILSCALE, (p, ip)
            except ValueError:  # not an address, e.g. a version number
                pass


# (c)
@pytest.mark.parametrize("mutate", [
    lambda c: setattr(c.reasoning, "ollama_url", "http://100.97.67.75:11434"),
    lambda c: setattr(c.perception.vision, "ollama_url", "http://10.0.0.2:11434"),
    lambda c: (setattr(c.reasoning, "fallback_model", "gemma4:e2b"),
               setattr(c.reasoning, "fallback_url", "http://gpu.example.com:11434")),
], ids=["reasoning", "vision", "fallback"])
def test_enforce_local_only_rejects_remote_endpoints(mutate):
    cfg = JarvisConfig()
    mutate(cfg)
    with pytest.raises(ConfigError):
        enforce_local_only(cfg)
    cfg.runtime.local_only = False
    enforce_local_only(cfg)


def test_remote_fallback_url_is_ignored_while_fallback_disabled():
    cfg = JarvisConfig()
    cfg.reasoning.fallback_url = "http://gpu.example.com:11434"
    enforce_local_only(cfg)


def test_load_config_refuses_remote_url_unless_local_only_is_off(tmp_path: Path):
    body = '[reasoning]\nollama_url = "http://100.97.67.75:11434"\n'
    path = tmp_path / "remote.toml"
    path.write_text(body)
    with pytest.raises(ConfigError):
        load_config(path)
    path.write_text("[runtime]\nlocal_only = false\n" + body)
    assert load_config(path).reasoning.ollama_url == "http://100.97.67.75:11434"


# (d)
_CLOUD_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(openai|anthropic|google\.generativeai|google\.genai|cohere|mistralai)\b", re.MULTILINE
)
_CLOUD_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com")


def test_no_cloud_llm_client_in_app():
    for p in (ROOT / "app").rglob("*.py"):
        text = p.read_text()
        assert not _CLOUD_IMPORT.search(text), p
        assert not any(h in text for h in _CLOUD_HOSTS), p


# (e)
def test_default_model_is_plain_ollama_without_fallback():
    from app.reasoning import build_reasoning_model
    from app.reasoning.fallback import FallbackModel
    from app.reasoning.local_model import OllamaModel

    model = build_reasoning_model(load_config().reasoning)
    try:
        assert type(model) is OllamaModel
        assert not isinstance(model, FallbackModel)
    finally:
        model.close()


def test_local_server_down_fails_explicitly_and_only_dials_loopback(monkeypatch):
    from app.reasoning import build_reasoning_model
    from app.reasoning.model import ReasoningError

    dialed = []
    real_connect = socket.socket.connect

    def connect(self, address):
        dialed.append(address)
        return real_connect(self, address)

    # httpcore dials via socket.create_connection, which calls socket.socket.connect.
    monkeypatch.setattr(socket.socket, "connect", connect)

    cfg = dataclasses.replace(load_config().reasoning, ollama_url="http://127.0.0.1:1")
    model = build_reasoning_model(cfg)
    try:
        with pytest.raises(ReasoningError):
            list(model.stream([{"role": "user", "content": "hi"}]))
    finally:
        model.close()
    assert dialed, "expected a connection attempt"
    assert all(ipaddress.ip_address(a[0]).is_loopback for a in dialed), dialed


def test_onnxruntime_telemetry_is_disabled_by_importing_app() -> None:
    import os

    import app  # noqa: F401

    assert os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
