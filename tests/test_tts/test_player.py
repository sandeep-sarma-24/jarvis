"""Tests for LocalAudioPlayer — playback + interruption."""
import threading
import time

import numpy as np
import pytest

from app.tts.player import LocalAudioPlayer


def _silence(seconds: float = 0.5, sample_rate: int = 24000) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


class TestPlayerBasics:
    def test_not_playing_initially(self):
        player = LocalAudioPlayer()
        assert player.is_playing() is False
        player.close()

    def test_stop_when_not_playing_is_safe(self):
        player = LocalAudioPlayer()
        player.stop()
        player.stop()
        player.stop()
        assert player.is_playing() is False
        player.close()

    def test_repeated_stop_no_crash(self):
        player = LocalAudioPlayer()
        for _ in range(20):
            player.stop()
        player.close()

    def test_close_when_not_playing(self):
        player = LocalAudioPlayer()
        player.close()


@pytest.mark.hardware
class TestPlayerPlayback:
    """Requires audio hardware. Skip in CI."""

    def test_play_starts_playback(self):
        player = LocalAudioPlayer()
        audio = _silence(0.3)
        player.play(audio, 24000)
        time.sleep(0.05)  # give thread time to start
        assert player.is_playing() is True
        player.stop()
        player.close()

    def test_play_then_stop_interrupts(self):
        player = LocalAudioPlayer()
        audio = _silence(2.0)  # 2 seconds
        player.play(audio, 24000)
        time.sleep(0.05)
        player.stop()
        time.sleep(0.1)
        assert player.is_playing() is False
        player.close()

    def test_new_play_stops_previous(self):
        player = LocalAudioPlayer()
        audio1 = _silence(2.0)
        audio2 = _silence(0.2)
        player.play(audio1, 24000)
        time.sleep(0.05)
        player.play(audio2, 24000)
        time.sleep(0.4)
        assert player.is_playing() is False
        player.close()

    def test_volume_scaling(self):
        player = LocalAudioPlayer(volume=0.5)
        audio = np.ones(24000, dtype=np.float32) * 0.8
        player.play(audio, 24000)
        time.sleep(0.05)
        player.stop()
        player.close()

    def test_playback_completes_naturally(self):
        player = LocalAudioPlayer()
        audio = _silence(0.1)  # very short
        player.play(audio, 24000)
        time.sleep(0.3)  # wait for completion
        assert player.is_playing() is False
        player.close()


class TestPlayerThreadSafety:
    def test_concurrent_stop_calls(self):
        player = LocalAudioPlayer()
        threads = [
            threading.Thread(target=player.stop)
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        player.close()
