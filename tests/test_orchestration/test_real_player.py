"""TurnManager with the real StreamingPlayer (sounddevice mocked, device callback driven by hand)."""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from app.config import JarvisConfig
from app.context.builder import ContextBuilder
from app.contracts import LatencyRecorder
from app.orchestration.turns import TurnManager
from app.tts.model import AudioChunk, TTSStreamStats
from app.tts.player import StreamingPlayer

from tests.test_orchestration.fakes import (
    FakeAttention, FakeBus, FakeFrontend, StreamingModel, transcript, wait_for,
)

SR = 1000
OLD, NEW = 1.0, 2.0


class ValueTTS:
    """Each sentence becomes 4 chunks of 50 samples; value marks old vs new turn."""

    model_name = "value-tts"
    sample_rate = SR

    def load(self):
        pass

    def close(self):
        pass

    async def stream_to(self, text, sink, *, cancel=None):
        value = OLD if text.startswith("Old") else NEW

        def run():
            for _ in range(4):
                if cancel is not None and cancel.is_set():
                    return
                time.sleep(0.002)
                if not sink(AudioChunk(audio=np.full(50, value, np.float32), sample_rate=SR)):
                    return

        await asyncio.to_thread(run)
        return TTSStreamStats(first_chunk_ms=1.0, synth_ms=1.0, audio_seconds=0.2, chunks=4, cancelled=False)


async def test_no_sample_of_superseded_turn_reaches_device():
    with patch("app.tts.player.sd.OutputStream") as os_cls:
        os_cls.return_value = MagicMock()
        holder = {}
        player = StreamingPlayer(
            max_queue_seconds=0.3,
            on_first_audio=lambda t: holder["tm"].on_first_audio(t),
            on_done=lambda t, o: holder["tm"].on_playback_done(t, o),
        )
        player.start(SR)
        callback = os_cls.call_args.kwargs["callback"]

        pulls: list[tuple[float, np.ndarray]] = []
        stop_device = threading.Event()

        def device():
            while not stop_device.is_set():
                started = time.time()
                out = np.zeros((20, 1), np.float32)
                callback(out, 20, None, None)
                pulls.append((started, out[:, 0].copy()))
                time.sleep(0.002)

        dev = threading.Thread(target=device, daemon=True)
        dev.start()

        frontend = FakeFrontend()
        recorder = LatencyRecorder()
        tm = TurnManager(
            cfg=JarvisConfig(), bus=FakeBus(), attention=FakeAttention(),
            context_builder=ContextBuilder(),
            model=StreamingModel([f"Old sentence number {i}. " for i in range(40)], delay=0.01),
            tts=ValueTTS(), player=player, frontend=frontend, recorder=recorder,
        )
        holder["tm"] = tm
        try:
            await tm.handle_transcript(transcript("jarvis tell a long story"))
            await wait_for(lambda: any((p == OLD).any() for _, p in pulls))

            tm._model = StreamingModel(["New answer number one. ", "New answer number two. "])
            await tm.handle_transcript(transcript("jarvis something else"))
            superseded_at = time.time()
            await asyncio.wait_for(asyncio.shield(tm._last_task), 3.0)
        finally:
            stop_device.set()
            dev.join(1.0)
            player.close()

        after = [p for started, p in pulls if started > superseded_at]
        assert after and not any((p == OLD).any() for p in after)
        assert any((p == NEW).any() for p in after)
        assert [r["outcome"] for r in recorder.turns] == ["cancelled", "completed"]
        assert frontend.states[-1] is False
        assert "first_audio" in recorder.turns[-1] and "playback_done" in recorder.turns[-1]
