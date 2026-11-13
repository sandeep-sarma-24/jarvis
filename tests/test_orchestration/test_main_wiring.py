"""app.main.run() startup/shutdown order with every hardware and model component faked."""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app import main as app_main
from app.config import JarvisConfig


class Log(list):
    def mark(self, name):
        self.append(name)


def _fakes(log: Log, *, fail_load=False, frontend_seconds=0.0):
    class Mic:
        def __init__(self, cfg):
            class Buf:
                items = [1, 2, 3]

                def get_nowait(self):
                    return self.items.pop() if self.items else None
            self.buffer = Buf()

        def start(self, loop):
            log.mark("mic.start")

        def stop(self):
            log.mark("mic.stop")

    class VAD:
        def __init__(self, threshold):
            pass

        def load(self):
            log.mark("vad.load")

    class ASR:
        def __init__(self, cfg):
            pass

        def load(self):
            if fail_load:
                raise RuntimeError("asr load failed")
            log.mark("asr.load")

        def close(self):
            log.mark("asr.close")

    class Frontend:
        def __init__(self, mic, vad, seg, asr, *, on_transcript, on_barge_in, barge_in, playback_tail_ms):
            self.metrics = {}
            assert on_barge_in is None  # barge-in disabled by default

        async def run(self):
            log.mark(f"frontend.run drained={mic_holder.buffer.items == []}")
            await asyncio.sleep(frontend_seconds)

        def set_playback_active(self, active):
            pass

    class Model:
        model_name = "m"

        def warm_up(self):
            log.mark("model.warm_up")

        def close(self):
            log.mark("model.close")

    class TTS:
        sample_rate = 24000

        def __init__(self, **kw):
            pass

        def load(self):
            log.mark("tts.load")

        def close(self):
            log.mark("tts.close")

    class Player:
        def __init__(self, **kw):
            self.stats = {}

        def start(self, sr):
            log.mark(f"player.start {sr}")

        def stop(self):
            pass

        def close(self):
            log.mark("player.close")

    mic_holder = None

    def make_mic(cfg):
        nonlocal mic_holder
        mic_holder = Mic(cfg)
        return mic_holder

    return [
        patch("app.audio.capture.MicrophoneCapture", make_mic),
        patch("app.audio.vad.VoiceActivityDetector", VAD),
        patch("app.audio.asr.SpeechRecognizer", ASR),
        patch("app.audio.frontend.AudioFrontend", Frontend),
        patch("app.reasoning.build_reasoning_model", lambda cfg: Model()),
        patch("app.tts.local_tts.PhononTTS", TTS),
        patch("app.tts.player.StreamingPlayer", Player),
    ]


def _cfg():
    cfg = JarvisConfig()
    cfg.memory.enabled = False
    cfg.perception.enabled = False
    cfg.ui.enabled = False
    return cfg


async def _run(log, **kw):
    patches = _fakes(log, **kw)
    for p in patches:
        p.start()
    try:
        with patch.object(asyncio.get_running_loop(), "add_signal_handler", lambda *a: log.mark("signals")):
            await asyncio.wait_for(app_main.run(_cfg()), 5.0)
    finally:
        for p in patches:
            p.stop()


async def test_startup_order_and_reverse_shutdown():
    log = Log()
    await _run(log)
    await asyncio.sleep(0.05)  # warm-up daemon thread
    assert log[0] == "signals"
    i = log.index
    assert i("mic.start") < min(i("vad.load"), i("asr.load"), i("tts.load"))
    assert i("tts.load") < i("player.start 24000")
    assert i("player.start 24000") < i("frontend.run drained=True")
    assert "model.warm_up" in log
    # reverse order: mic stops before the player, TTS and model are closed
    assert i("mic.stop") < i("asr.close") < i("player.close") < i("tts.close") < i("model.close")


async def test_failed_load_still_releases_mic_and_closes_components():
    log = Log()
    with pytest.raises(RuntimeError, match="asr load failed"):
        await _run(log, fail_load=True)
    assert "mic.stop" in log and "player.close" in log and "model.close" in log
    assert not any(e.startswith("frontend.run") for e in log)


async def test_programmatic_remote_config_is_refused_before_anything_opens():
    from app.config import ConfigError

    cfg = _cfg()
    cfg.reasoning.ollama_url = "http://192.168.1.5:11434"
    log = Log()
    with patch("app.audio.capture.MicrophoneCapture", lambda c: log.mark("mic")):
        with pytest.raises(ConfigError):
            await app_main.run(cfg)
    assert log == []


def test_unreachable_reasoning_host_logs_error_with_remedy(caplog):
    class Down:
        def warm_up(self):
            return False

    with caplog.at_level("ERROR", logger="jarvis"):
        app_main._warm_up(Down(), JarvisConfig())
    msg = caplog.records[-1].getMessage()
    assert "http://127.0.0.1:11434" in msg and "ollama serve" in msg and "ollama pull gemma4:e2b-it-qat" in msg
