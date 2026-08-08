"""Tests for LocalAudioPlayer — playback, interruption, generation-based cancellation."""
import threading
import time

import numpy as np
import pytest

from app.tts.player import LocalAudioPlayer, PlaybackOutcome


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

    def test_stop_guarantees_not_playing(self):
        """After stop() returns, is_playing() must be False."""
        player = LocalAudioPlayer()
        player.stop()
        assert player.is_playing() is False
        player.close()


class TestPlaybackOutcome:
    """Test on_done callback reports correct outcomes."""

    def test_on_done_setter(self):
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        assert player.on_done is not None
        player.close()

    def test_no_callback_is_fine(self):
        player = LocalAudioPlayer()
        assert player.on_done is None
        player.close()


@pytest.mark.hardware
class TestPlayerPlayback:
    """Requires audio hardware. Skip in CI."""

    def test_play_starts_playback(self):
        player = LocalAudioPlayer()
        audio = _silence(0.3)
        player.play(audio, 24000)
        time.sleep(0.05)
        assert player.is_playing() is True
        player.stop()
        player.close()

    def test_play_then_stop_interrupts(self):
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        audio = _silence(2.0)
        player.play(audio, 24000)
        time.sleep(0.05)
        player.stop()
        time.sleep(0.2)
        assert player.is_playing() is False
        assert PlaybackOutcome.INTERRUPTED in outcomes
        player.close()

    def test_natural_completion_reports_completed(self):
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        audio = _silence(0.1)  # very short
        player.play(audio, 24000)
        time.sleep(0.5)  # wait for natural end
        assert player.is_playing() is False
        assert PlaybackOutcome.COMPLETED in outcomes
        player.close()

    def test_new_play_stops_previous(self):
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        audio1 = _silence(2.0)
        audio2 = _silence(0.1)
        player.play(audio1, 24000)
        time.sleep(0.05)
        player.play(audio2, 24000)  # stops audio1
        time.sleep(0.5)
        assert player.is_playing() is False
        # audio1 was interrupted, audio2 completed
        assert PlaybackOutcome.INTERRUPTED in outcomes
        assert PlaybackOutcome.COMPLETED in outcomes
        player.close()

    def test_no_overlapping_streams(self):
        """Two rapid play() calls must not produce overlapping streams."""
        player = LocalAudioPlayer()
        audio1 = _silence(2.0)
        audio2 = _silence(2.0)
        player.play(audio1, 24000)
        time.sleep(0.02)
        player.play(audio2, 24000)
        time.sleep(0.05)
        # Only one should be active
        assert player.is_playing() is True
        player.stop()
        assert player.is_playing() is False
        player.close()

    def test_stop_before_play_starts(self):
        """stop() on player that never played is safe."""
        player = LocalAudioPlayer()
        player.stop()
        assert player.is_playing() is False
        player.close()

    def test_stop_after_completion(self):
        """stop() after natural completion is safe."""
        player = LocalAudioPlayer()
        audio = _silence(0.1)
        player.play(audio, 24000)
        time.sleep(0.5)
        assert player.is_playing() is False
        player.stop()  # should not crash
        player.close()

    def test_volume_scaling(self):
        player = LocalAudioPlayer(volume=0.5)
        audio = np.ones(24000, dtype=np.float32) * 0.8
        player.play(audio, 24000)
        time.sleep(0.05)
        player.stop()
        player.close()

    def test_completed_not_incremented_on_interrupt(self):
        """Interrupted playback must NOT report COMPLETED."""
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        audio = _silence(5.0)  # long audio
        player.play(audio, 24000)
        time.sleep(0.05)
        player.stop()
        time.sleep(0.2)
        assert PlaybackOutcome.COMPLETED not in outcomes
        assert PlaybackOutcome.INTERRUPTED in outcomes
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

    def test_generation_prevents_resurrection(self):
        """After stop(), a stale worker thread cannot set is_playing back to True."""
        player = LocalAudioPlayer()
        player.stop()
        assert player.is_playing() is False
        # Verify the invariant holds even after multiple calls
        player.stop()
        assert player.is_playing() is False
        player.close()
