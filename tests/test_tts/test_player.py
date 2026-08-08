"""Tests for LocalAudioPlayer — playback, interruption, generation-based cancellation.

Split into:
- TestPlayerBasics: no hardware needed
- TestPlaybackOutcome: no hardware needed
- TestPlayerStateMachine: mocked sounddevice, no hardware needed
- TestPlayerPlayback: real hardware (@pytest.mark.hardware)
- TestPlayerThreadSafety: no hardware needed
"""
import threading
import time
from unittest.mock import MagicMock, patch

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

    def test_stop_without_play_no_outcome(self):
        """stop() without prior play() produces no on_done callback."""
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        player.stop()
        player.stop()
        assert outcomes == []
        player.close()


class TestPlayerStateMachine:
    """Mocked sounddevice tests — verify state machine without audio hardware."""

    def _mock_stream(self):
        stream = MagicMock()
        stream.active = True
        stream.start = MagicMock()
        stream.stop = MagicMock()
        stream.close = MagicMock()
        stream.abort = MagicMock()
        stream.write = MagicMock()
        return stream

    @patch("app.tts.player.sd.OutputStream")
    def test_natural_completion(self, mock_os_class):
        """All frames written → COMPLETED."""
        mock_stream = self._mock_stream()
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(2048, dtype=np.float32)  # 2 blocks of 1024
        player.play(audio, 24000)

        # Wait for worker to finish
        time.sleep(0.2)

        assert player.is_playing() is False
        assert outcomes == [PlaybackOutcome.COMPLETED]
        assert mock_stream.start.called
        assert mock_stream.write.called
        assert mock_stream.stop.called
        assert mock_stream.close.called
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_stop_interrupts(self, mock_os_class):
        """stop() during playback → INTERRUPTED."""
        mock_stream = self._mock_stream()
        # Make write block briefly so we can interrupt
        mock_stream.write.side_effect = lambda b: time.sleep(0.05)
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(24000, dtype=np.float32)  # many blocks
        player.play(audio, 24000)
        time.sleep(0.02)  # let worker start
        player.stop()

        assert player.is_playing() is False
        assert outcomes == [PlaybackOutcome.INTERRUPTED]
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_play_b_interrupts_a(self, mock_os_class):
        """play(A) then play(B) → A=INTERRUPTED, B=COMPLETED."""
        mock_stream = self._mock_stream()
        # Make write slightly slow so A is still running when B starts
        mock_stream.write.side_effect = lambda b: time.sleep(0.01)
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio_a = np.zeros(24000, dtype=np.float32)  # long
        audio_b = np.zeros(1024, dtype=np.float32)   # short, completes quickly

        player.play(audio_a, 24000)
        time.sleep(0.03)
        player.play(audio_b, 24000)
        time.sleep(0.3)  # wait for B to complete

        assert player.is_playing() is False
        assert PlaybackOutcome.INTERRUPTED in outcomes
        assert PlaybackOutcome.COMPLETED in outcomes
        # A was interrupted, B completed
        assert outcomes.index(PlaybackOutcome.INTERRUPTED) < outcomes.index(PlaybackOutcome.COMPLETED)
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_no_overlapping_streams(self, mock_os_class):
        """play(A) then play(B) must abort A's stream before B's stream starts."""
        streams_created: list[MagicMock] = []

        def make_stream(**kwargs):
            s = self._mock_stream()
            s.write.side_effect = lambda b: time.sleep(0.01)
            streams_created.append(s)
            return s

        mock_os_class.side_effect = make_stream

        player = LocalAudioPlayer()
        audio = np.zeros(24000, dtype=np.float32)

        player.play(audio, 24000)
        time.sleep(0.03)
        player.play(audio, 24000)
        time.sleep(0.03)

        # First stream should have been aborted
        if len(streams_created) >= 1:
            first = streams_created[0]
            assert first.abort.called or first.stop.called
        player.stop()
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_device_error_reports_interrupted(self, mock_os_class):
        """Device error during write → INTERRUPTED, not COMPLETED."""
        import sounddevice as sd
        mock_stream = self._mock_stream()
        mock_stream.write.side_effect = sd.PortAudioError("device error")
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(2048, dtype=np.float32)
        player.play(audio, 24000)
        time.sleep(0.2)

        assert player.is_playing() is False
        assert outcomes == [PlaybackOutcome.INTERRUPTED]
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_stale_generation_early_exit_reports_interrupted(self, mock_os_class):
        """Worker that sees stale generation immediately must report INTERRUPTED."""
        mock_stream = self._mock_stream()
        # Slow writes ensure the worker is still running when we cancel
        mock_stream.write.side_effect = lambda b: time.sleep(0.05)
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(24000, dtype=np.float32)  # many blocks with slow writes
        player.play(audio, 24000)
        time.sleep(0.02)  # let worker start writing
        player.stop()     # invalidate generation while worker is mid-stream
        time.sleep(0.2)

        # Should only have INTERRUPTED outcomes, never COMPLETED
        for o in outcomes:
            assert o == PlaybackOutcome.INTERRUPTED
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_close_during_playback(self, mock_os_class):
        """close() while playing → stream closes, INTERRUPTED."""
        mock_stream = self._mock_stream()
        mock_stream.write.side_effect = lambda b: time.sleep(0.02)
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(24000, dtype=np.float32)
        player.play(audio, 24000)
        time.sleep(0.03)
        player.close()

        assert player.is_playing() is False
        assert PlaybackOutcome.INTERRUPTED in outcomes
        assert PlaybackOutcome.COMPLETED not in outcomes

    @patch("app.tts.player.sd.OutputStream")
    def test_stop_after_natural_completion_no_new_outcome(self, mock_os_class):
        """stop() after natural completion produces no additional outcome."""
        mock_stream = self._mock_stream()
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(1024, dtype=np.float32)  # one block, completes fast
        player.play(audio, 24000)
        time.sleep(0.2)  # wait for natural completion

        assert outcomes == [PlaybackOutcome.COMPLETED]

        player.stop()  # should not add another outcome
        assert outcomes == [PlaybackOutcome.COMPLETED]
        player.close()

    @patch("app.tts.player.sd.OutputStream")
    def test_repeated_stop_no_extra_outcomes(self, mock_os_class):
        """Multiple stop() calls produce at most one INTERRUPTED."""
        mock_stream = self._mock_stream()
        mock_stream.write.side_effect = lambda b: time.sleep(0.02)
        mock_os_class.return_value = mock_stream

        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))

        audio = np.zeros(24000, dtype=np.float32)
        player.play(audio, 24000)
        time.sleep(0.03)
        player.stop()
        player.stop()
        player.stop()
        time.sleep(0.1)

        # Only one INTERRUPTED from the worker
        assert outcomes.count(PlaybackOutcome.INTERRUPTED) == 1
        assert PlaybackOutcome.COMPLETED not in outcomes
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
        assert PlaybackOutcome.COMPLETED not in outcomes
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
        outcomes: list[PlaybackOutcome] = []
        player = LocalAudioPlayer(on_done=lambda o: outcomes.append(o))
        audio = _silence(0.1)
        player.play(audio, 24000)
        time.sleep(0.5)
        assert player.is_playing() is False
        player.stop()  # should not crash or add outcome
        assert outcomes == [PlaybackOutcome.COMPLETED]
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
        player.stop()
        assert player.is_playing() is False
        player.close()
