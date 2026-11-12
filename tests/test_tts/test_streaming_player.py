"""StreamingPlayer with sounddevice mocked; the device callback is driven by hand."""
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.tts.model import AudioChunk, StreamingAudioPlayer
from app.tts.player import PlaybackOutcome, StreamingPlayer

SR = 1000  # small rate keeps sample arithmetic readable


def _chunk(value: float, n: int = 100, sr: int = SR) -> AudioChunk:
    return AudioChunk(audio=np.full(n, value, dtype=np.float32), sample_rate=sr)


@pytest.fixture
def rig():
    """Yields (player, pull, events). pull(frames) runs one device callback."""
    events: list[tuple] = []
    with patch("app.tts.player.sd.OutputStream") as os_cls:
        os_cls.return_value = MagicMock()
        player = StreamingPlayer(
            max_queue_seconds=0.3,
            on_first_audio=lambda t: events.append(("first", t)),
            on_done=lambda t, o: events.append(("done", t, o)),
        )
        player.start(SR)
        callback = os_cls.call_args.kwargs["callback"]

        def pull(frames: int = 100) -> np.ndarray:
            out = np.full((frames, 1), 9.0, dtype=np.float32)  # garbage the callback must overwrite
            callback(out, frames, None, None)
            return out[:, 0].copy()

        yield player, pull, events, os_cls
        player.close()


def _feed_in_thread(player, chunk, turn):
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ok", player.feed(chunk, turn)), daemon=True)
    t.start()
    return t, result


def test_conforms_and_opens_one_stream(rig):
    player, _, _, os_cls = rig
    assert isinstance(player, StreamingAudioPlayer)
    player.start(SR)  # idempotent
    assert os_cls.call_count == 1
    kw = os_cls.call_args.kwargs
    assert kw["channels"] == 1 and kw["dtype"] == "float32" and kw["samplerate"] == SR


def test_chunks_play_in_order(rig):
    player, pull, events, _ = rig
    player.begin("a")
    for v in (1, 2, 3):
        assert player.feed(_chunk(v), "a")
    assert list(pull(150)) == [1] * 100 + [2] * 50
    assert list(pull(150)) == [2] * 50 + [3] * 100
    assert events == [("first", "a")]


def test_feed_for_non_current_turn_is_rejected(rig):
    player, pull, _, _ = rig
    player.begin("a")
    assert player.feed(_chunk(5), "b") is False
    assert not pull().any()
    assert player.stats["chunks_dropped_stale"] == 1


def test_begin_discards_old_audio_and_interrupts_old_turn(rig):
    player, pull, events, _ = rig
    player.begin("a")
    player.feed(_chunk(1), "a")
    player.begin("b")
    player.feed(_chunk(2), "b")
    assert list(pull()) == [2] * 100
    assert ("done", "a", PlaybackOutcome.INTERRUPTED) in events


def test_end_fires_completed_once_after_drain(rig):
    player, pull, events, _ = rig
    player.begin("a")
    player.feed(_chunk(1), "a")
    player.end("a")
    assert not any(e[0] == "done" for e in events)
    assert player.is_playing()
    pull()
    pull()
    player.stop()
    assert [e for e in events if e[0] == "done"] == [("done", "a", PlaybackOutcome.COMPLETED)]
    assert not player.is_playing()


def test_end_with_empty_queue_completes_immediately(rig):
    player, _, events, _ = rig
    player.begin("a")
    player.end("a")
    assert events == [("done", "a", PlaybackOutcome.COMPLETED)]


def test_stop_fires_interrupted_once_and_silences(rig):
    player, pull, events, _ = rig
    player.begin("a")
    player.feed(_chunk(1, 250), "a")
    player.stop()
    player.stop()
    player.end("a")
    assert not pull().any()
    assert events == [("done", "a", PlaybackOutcome.INTERRUPTED)]


def test_back_to_back_turns_never_complete_new_turn_from_old(rig):
    """The suppression race: the old turn's done must carry the old id only."""
    player, pull, events, _ = rig
    player.begin("a")
    player.feed(_chunk(1), "a")
    player.end("a")
    player.begin("b")  # before a drained
    player.feed(_chunk(2), "b")
    pull()
    dones = [e for e in events if e[0] == "done"]
    assert dones == [("done", "a", PlaybackOutcome.INTERRUPTED)]
    player.end("b")
    pull()
    dones = [e for e in events if e[0] == "done"]
    assert dones == [("done", "a", PlaybackOutcome.INTERRUPTED), ("done", "b", PlaybackOutcome.COMPLETED)]


def test_on_done_once_per_begin_over_many_turns(rig):
    player, pull, events, _ = rig
    for i in range(20):
        tid = f"t{i}"
        player.begin(tid)
        player.feed(_chunk(1, 50), tid)
        if i % 3 == 0:
            player.end(tid)
            pull()
        elif i % 3 == 1:
            player.stop()
    player.stop()
    dones = [e[1] for e in events if e[0] == "done"]
    assert sorted(dones) == sorted(f"t{i}" for i in range(20))


def test_feed_blocks_when_full_and_resumes_on_drain(rig):
    player, pull, _, _ = rig
    player.begin("a")
    for _ in range(3):  # 300 samples == max_queue_seconds
        assert player.feed(_chunk(1), "a")
    t, result = _feed_in_thread(player, _chunk(2), "a")
    t.join(0.1)
    assert t.is_alive()
    pull(100)
    t.join(1)
    assert result["ok"] is True
    assert player.stats["max_queue_seconds"] == pytest.approx(0.3)


def test_stop_wakes_blocked_feeder_quickly_and_stays_silent(rig):
    player, pull, events, _ = rig
    player.begin("a")
    for _ in range(3):
        player.feed(_chunk(1), "a")
    t, result = _feed_in_thread(player, _chunk(2), "a")
    t.join(0.05)
    assert t.is_alive()
    t0 = time.perf_counter()
    player.stop()
    assert (time.perf_counter() - t0) < 0.02
    t.join(1)
    assert result["ok"] is False
    for _ in range(5):
        assert not pull().any()
    assert player.stats["queue_seconds"] == 0


def test_underrun_outputs_silence_and_is_counted(rig):
    player, pull, _, _ = rig
    player.begin("a")
    assert not pull().any()  # waiting for first audio is not an underrun
    assert player.stats["underruns"] == 0
    player.feed(_chunk(1, 60), "a")
    out = pull(100)
    assert list(out) == [1] * 60 + [0] * 40
    assert not pull().any()
    assert player.stats["underruns"] == 2
    player.end("a")
    pull()
    assert player.stats["underruns"] == 2


def test_sample_rate_mismatch_raises(rig):
    player, _, _, _ = rig
    player.begin("a")
    with pytest.raises(ValueError):
        player.feed(_chunk(1, sr=SR * 2), "a")


def test_volume_applied():
    with patch("app.tts.player.sd.OutputStream") as os_cls:
        player = StreamingPlayer(volume=0.5)
        player.start(SR)
        cb = os_cls.call_args.kwargs["callback"]
        player.begin("a")
        player.feed(_chunk(0.8), "a")
        out = np.zeros((100, 1), dtype=np.float32)
        cb(out, 100, None, None)
        assert np.allclose(out, 0.4)
        player.close()
