"""Tests for TTS model abstractions and types."""
import numpy as np
import pytest

from app.tts.model import AudioChunk, AudioPlayer, TextToSpeech, TTSResult


class FakeTTS:
    """Minimal TTS implementation for testing."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-tts"

    def synthesize(self, text: str) -> TTSResult:
        self.calls.append(text)
        if self._fail:
            return TTSResult(
                success=False, audio=None, sample_rate=24000,
                channels=1, duration_seconds=0.0, latency_ms=0.1,
                model="fake-tts", error="forced failure",
            )
        samples = np.zeros(24000, dtype=np.float32)  # 1 second of silence
        return TTSResult(
            success=True, audio=samples, sample_rate=24000,
            channels=1, duration_seconds=1.0, latency_ms=0.5,
            model="fake-tts",
        )

    def synthesize_stream(self, text: str):
        self.calls.append(text)
        audio = np.zeros(12000, dtype=np.float32)
        yield AudioChunk(audio=audio, sample_rate=24000, is_last=False)
        yield AudioChunk(audio=audio, sample_rate=24000, is_last=True)

    def close(self) -> None:
        pass


class FakePlayer:
    """Minimal AudioPlayer for testing with outcome tracking."""

    def __init__(self) -> None:
        self.plays: list[tuple[int, int]] = []  # (num_samples, sample_rate)
        self._playing = False
        self.stop_count = 0
        self.outcomes: list[str] = []  # "completed" or "interrupted"

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if self._playing:
            self.outcomes.append("interrupted")
        self.plays.append((len(audio), sample_rate))
        self._playing = True

    def stop(self) -> None:
        if self._playing:
            self.outcomes.append("interrupted")
        self._playing = False
        self.stop_count += 1

    def is_playing(self) -> bool:
        return self._playing

    def complete(self) -> None:
        """Simulate natural completion."""
        if self._playing:
            self._playing = False
            self.outcomes.append("completed")

    def close(self) -> None:
        self.stop()


class TestTTSResult:
    def test_success_result(self):
        audio = np.zeros(24000, dtype=np.float32)
        r = TTSResult(
            success=True, audio=audio, sample_rate=24000,
            channels=1, duration_seconds=1.0, latency_ms=50.0,
            model="test",
        )
        assert r.success is True
        assert r.audio is not None
        assert r.error is None

    def test_failure_result(self):
        r = TTSResult(
            success=False, audio=None, sample_rate=24000,
            channels=1, duration_seconds=0.0, latency_ms=1.0,
            model="test", error="something broke",
        )
        assert r.success is False
        assert r.audio is None
        assert r.error == "something broke"

    def test_frozen(self):
        r = TTSResult(
            success=True, audio=np.zeros(10, dtype=np.float32),
            sample_rate=24000, channels=1, duration_seconds=0.01,
            latency_ms=1.0, model="test",
        )
        with pytest.raises(Exception):
            r.success = False  # type: ignore


class TestAudioChunk:
    def test_fields(self):
        audio = np.ones(1024, dtype=np.float32)
        c = AudioChunk(audio=audio, sample_rate=24000, is_last=True)
        assert c.is_last is True
        assert c.sample_rate == 24000
        assert len(c.audio) == 1024


class TestProtocolConformance:
    def test_fake_tts_conforms_to_protocol(self):
        tts = FakeTTS()
        assert isinstance(tts, TextToSpeech)

    def test_fake_player_conforms_to_protocol(self):
        player = FakePlayer()
        assert isinstance(player, AudioPlayer)


class TestFakeTTS:
    def test_synthesize_success(self):
        tts = FakeTTS()
        result = tts.synthesize("Hello world")
        assert result.success is True
        assert result.audio is not None
        assert result.sample_rate == 24000
        assert len(tts.calls) == 1

    def test_synthesize_failure(self):
        tts = FakeTTS(fail=True)
        result = tts.synthesize("Hello world")
        assert result.success is False
        assert result.audio is None
        assert result.error is not None

    def test_stream_yields_chunks(self):
        tts = FakeTTS()
        chunks = list(tts.synthesize_stream("Hello world"))
        assert len(chunks) == 2
        assert chunks[-1].is_last is True


class TestFakePlayer:
    def test_play_sets_playing(self):
        player = FakePlayer()
        assert player.is_playing() is False
        player.play(np.zeros(100, dtype=np.float32), 24000)
        assert player.is_playing() is True

    def test_stop_clears_playing(self):
        player = FakePlayer()
        player.play(np.zeros(100, dtype=np.float32), 24000)
        player.stop()
        assert player.is_playing() is False

    def test_stop_when_not_playing_is_safe(self):
        player = FakePlayer()
        player.stop()
        player.stop()
        player.stop()
        assert player.stop_count == 3
        # No outcomes generated when not playing
        assert player.outcomes == []

    def test_repeated_stop_no_crash(self):
        player = FakePlayer()
        player.play(np.zeros(100, dtype=np.float32), 24000)
        for _ in range(10):
            player.stop()
        assert player.is_playing() is False
        # Only one interruption (the first stop while playing)
        assert player.outcomes == ["interrupted"]

    def test_natural_completion(self):
        player = FakePlayer()
        player.play(np.zeros(100, dtype=np.float32), 24000)
        player.complete()
        assert player.is_playing() is False
        assert player.outcomes == ["completed"]

    def test_interruption_tracked(self):
        player = FakePlayer()
        player.play(np.zeros(100, dtype=np.float32), 24000)
        player.stop()
        assert player.outcomes == ["interrupted"]

    def test_new_play_interrupts_previous(self):
        player = FakePlayer()
        audio = np.zeros(100, dtype=np.float32)
        player.play(audio, 24000)
        player.play(audio, 24000)  # interrupts first
        assert player.outcomes == ["interrupted"]
        assert len(player.plays) == 2

    def test_complete_then_stop_no_double_outcome(self):
        """complete() then stop() must NOT produce both outcomes."""
        player = FakePlayer()
        player.play(np.zeros(100, dtype=np.float32), 24000)
        player.complete()
        player.stop()  # already inactive, should produce nothing
        assert player.outcomes == ["completed"]

    def test_play_a_play_b_complete_b(self):
        """play(A) → play(B) → complete(B) = A interrupted, B completed."""
        player = FakePlayer()
        audio = np.zeros(100, dtype=np.float32)
        player.play(audio, 24000)  # A
        player.play(audio, 24000)  # B interrupts A
        player.complete()  # B completes naturally
        assert player.outcomes == ["interrupted", "completed"]

    def test_stop_after_complete_no_outcome(self):
        """stop() after natural completion produces no new outcome."""
        player = FakePlayer()
        player.play(np.zeros(100, dtype=np.float32), 24000)
        player.complete()
        player.stop()
        player.stop()
        player.stop()
        assert player.outcomes == ["completed"]
