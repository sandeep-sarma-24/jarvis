#!/usr/bin/env python3
"""End-to-end check of the complete local pipeline, one PASS/FAIL line per check.

WAV (or --mic) -> real AudioFrontend (Silero VAD + SpeechSegmenter + Granite ASR, as in app.main)
-> TurnManager (attention, temp SQLite memory, context) -> local Ollama reasoning -> Phonon TTS
-> player (null sink by default, the real StreamingPlayer with --speaker).

Then a long-answer turn is barged in on after first audio and a third turn must play with no
stale audio from the cancelled one. Every outbound socket connect made from Python is recorded
and any non-loopback destination fails the run; HF_HUB_OFFLINE=1 is set before model imports.

Not a pytest module: run it directly (pytest's testpaths is tests/, and this file defines no
test functions). Exit code 0 only if every check passes.

Usage:
  python -m scripts.test_local_e2e [--wav PATH] [--expect capital,france] [--mic] [--speaker]
  python -m scripts.test_local_e2e --make-fixture     # (re)write the TTS-synthesized WAV fixtures
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
# Synthesized by the local Phonon TTS, not human recordings (hence the tts_ prefix).
FIXTURES = {
    "tts_capital_of_france_16k.wav": "Jarvis, what is the capital of France?",
    "tts_long_answer_question_16k.wav": "Jarvis, tell me the full history of the Roman Empire in detail.",
    # Used by scripts/benchmark_local_stack.py so its ASR input needs no TTS in-process.
    "tts_build_server_16k.wav": "Jarvis, what is the build server called?",
    "tts_focus_tips_16k.wav": "Jarvis, give me two tips for staying focused.",
}
FIRST, LONG = "tts_capital_of_france_16k.wav", "tts_long_answer_question_16k.wav"
SEED_FACT = "My sister lives in Lyon, France."
SEED_KEYWORD = "lyon"
SR, CHUNK = 16000, 512
TURN_TIMEOUT_S = 90.0


# ── Outbound connection recorder ────────────────────────────────────────

def is_local_destination(address) -> bool:
    """True for AF_UNIX paths and loopback hosts. Unresolved names other than localhost are not local."""
    if isinstance(address, (str, bytes)):
        return True
    host = address[0] if isinstance(address, tuple) and address else address
    if isinstance(host, bytes):
        host = host.decode()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


class ConnectionRecorder:
    """Wraps socket.socket.connect / connect_ex (httpx -> httpcore -> socket.create_connection uses them)."""

    def __init__(self) -> None:
        self.destinations: list[tuple[int, object]] = []
        self._orig: dict[str, object] = {}

    def install(self) -> None:
        rec = self
        for name in ("connect", "connect_ex"):
            orig = getattr(socket.socket, name)
            self._orig[name] = orig

            def wrapper(sock, address, _orig=orig):
                rec.destinations.append((sock.family, address))
                return _orig(sock, address)
            setattr(socket.socket, name, wrapper)

    def uninstall(self) -> None:
        for name, orig in self._orig.items():
            setattr(socket.socket, name, orig)

    @property
    def remote(self) -> list[object]:
        return [a for _, a in self.destinations if not is_local_destination(a)]


def lsof_remote_tcp(pid: int) -> list[str]:
    """Non-loopback TCP endpoints held by `pid` right now (catches native code, e.g. Rust, too)."""
    out = subprocess.run(["lsof", "-a", "-p", str(pid), "-i", "TCP", "-n", "-P"], capture_output=True, text=True).stdout
    bad = []
    for line in out.splitlines()[1:]:
        name = line.split()[8] if len(line.split()) > 8 else ""
        peer = name.split("->")[-1]
        host = peer.rsplit(":", 1)[0].strip("[]")
        if host != "*" and not is_local_destination((host,)):
            bad.append(name)
    return bad


def watch_remote_tcp(pid: int, seen: dict, stop: threading.Event, interval_s: float = 0.5) -> threading.Thread:
    """Poll lsof for the whole run: native code (onnxruntime, Rust) bypasses the socket wrapper."""
    t0 = time.monotonic()

    def loop() -> None:
        while not stop.is_set():
            for conn in lsof_remote_tcp(pid):
                seen.setdefault(conn, round(time.monotonic() - t0, 1))
            stop.wait(interval_s)
    th = threading.Thread(target=loop, name="lsof-watch", daemon=True)
    th.start()
    return th


# ── Reporting ──────────────────────────────────────────────────────────

class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", flush=True)
        return bool(ok)

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(r[1] for r in self.results)


# ── Fixtures ───────────────────────────────────────────────────────────

def write_wav16(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


async def make_fixtures() -> None:
    from app.config import load_config
    from app.tts.local_tts import PhononTTS
    from scripts.benchmark_local_stack import synth_clip_16k

    t = load_config().tts
    tts = PhononTTS(model_id=t.model, voice=t.voice, quant=t.quant)
    await asyncio.to_thread(tts.load)
    try:
        for name, text in FIXTURES.items():
            path = DATA / name
            write_wav16(path, await synth_clip_16k(tts, text))
            print(f"wrote {path} ({path.stat().st_size // 1024} KB, TTS-synthesized: {text!r})")
    finally:
        tts.close()


class WavMic:
    """Stands in for MicrophoneCapture: the AudioFrontend reads `buffer`, we fill it at real-time pace."""

    def __init__(self, loop) -> None:
        from app.audio.capture import BoundedAudioBuffer

        self.buffer = BoundedAudioBuffer(max_chunks=int(30 * SR / CHUNK))
        self.buffer.bind_loop(loop)
        self.suppressed = False


async def play_into(buffer, audio: np.ndarray, silence_s: float = 1.0) -> None:
    """Push audio plus trailing silence into the buffer in 512-sample chunks, one chunk per 32 ms."""
    audio = np.concatenate([np.zeros(int(0.3 * SR), np.float32), audio, np.zeros(int(silence_s * SR), np.float32)])
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        buffer.put(audio[i:i + CHUNK].copy(), time.time())
        await asyncio.sleep(CHUNK / SR)


async def _until(pred, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


class RecordingModel:
    """Delegates to the real model; records the messages it was given and the deltas per stream."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[list[dict]] = []
        self.deltas: list[int] = []

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @property
    def last_stats(self):
        return self.inner.last_stats

    def generate(self, *a, **k):
        return self.inner.generate(*a, **k)

    def stream(self, messages, **kwargs):
        self.calls.append(messages)
        self.deltas.append(0)
        for d in self.inner.stream(messages, **kwargs):
            self.deltas[-1] += 1
            yield d

    def warm_up(self):
        return self.inner.warm_up()

    def close(self) -> None:
        self.inner.close()


# ── The run ────────────────────────────────────────────────────────────

async def run(args, check: Checks, recorder_net: ConnectionRecorder) -> None:
    from app.audio.asr import SpeechRecognizer
    from app.audio.frontend import AudioFrontend
    from app.audio.segmentation import SpeechSegmenter
    from app.audio.vad import VoiceActivityDetector
    from app.config import is_loopback, load_config
    from app.events.bus import EventBus
    from app.memory.manager import MemoryManager
    from app.memory.sqlite_store import SQLiteStore
    from app.memory.types import MemoryType
    from app.orchestration.turns import TurnManager
    from app.reasoning import build_reasoning_model
    from app.tts.local_tts import PhononTTS
    from scripts.benchmark_audio import load_wav
    from scripts.benchmark_e2e import _attention_and_context
    from scripts.e2e_fakes import CountingRecorder, FakePlayer

    loop = asyncio.get_running_loop()
    cfg = load_config()
    cfg.perception.enabled = False
    r = cfg.reasoning
    print(f"reasoning: model={r.model} url={r.ollama_url} (loopback={is_loopback(r.ollama_url)})")
    wav1 = Path(args.wav) if args.wav else DATA / FIRST
    for p in (wav1, DATA / LONG):
        if not p.exists():
            check("fixtures present", False, f"{p} missing; run with --make-fixture")
            return

    mic = None
    if args.mic:
        from app.audio.capture import MicrophoneCapture

        mic = MicrophoneCapture(cfg.audio)
        mic.start(loop)  # before the MLX loads: PortAudio device conflict otherwise (see app.main)
    else:
        mic = WavMic(loop)

    vad = VoiceActivityDetector(threshold=cfg.vad.threshold)
    asr = SpeechRecognizer(cfg.asr)
    tts = PhononTTS(model_id=cfg.tts.model, voice=cfg.tts.voice, quant=cfg.tts.quant)
    t0 = time.perf_counter()
    try:
        await asyncio.gather(asyncio.to_thread(vad.load), asyncio.to_thread(asr.load), asyncio.to_thread(tts.load))
    except Exception as exc:
        check("models load with HF_HUB_OFFLINE=1", False, f"{type(exc).__name__}: {exc}")
        raise
    check("models load with HF_HUB_OFFLINE=1", True, f"VAD+ASR+TTS in {time.perf_counter() - t0:.1f} s")
    model = RecordingModel(build_reasoning_model(cfg.reasoning))
    check("Ollama warm-up", model.warm_up() is not False, f"{r.model} at {r.ollama_url}")

    first_audio: dict[str, float] = {}
    feeds: list[tuple[str, float]] = []  # (turn_id, time) of every chunk the player accepted
    tm = None
    if args.speaker:
        from app.tts.player import StreamingPlayer

        player = StreamingPlayer(volume=cfg.playback.volume, max_queue_seconds=cfg.playback.max_queue_seconds,
                                 on_first_audio=lambda tid: (first_audio.setdefault(tid, time.time()),
                                                             tm.on_first_audio(tid)),
                                 on_done=lambda tid, o: tm.on_playback_done(tid, o))
    else:
        player = FakePlayer(volume=cfg.playback.volume, max_queue_seconds=cfg.playback.max_queue_seconds)
        player.on_first_audio = lambda tid: (first_audio.setdefault(tid, time.time()), tm.on_first_audio(tid))
        player.on_done = lambda tid, o: tm.on_playback_done(tid, o)
    real_feed = player.feed

    def feed(chunk, turn_id):
        ok = real_feed(chunk, turn_id)
        if ok:
            feeds.append((turn_id, time.time()))
        return ok
    player.feed = feed
    await asyncio.to_thread(player.start, tts.sample_rate)

    tmp = tempfile.TemporaryDirectory()
    db = Path(tmp.name) / "jarvis.db"
    cfg.memory.database_path = str(db)
    memory = MemoryManager(SQLiteStore(db), cross_session=cfg.memory.cross_session)
    memory.start_session()
    memory.remember(SEED_FACT, MemoryType.FACT, importance=0.8)

    attention, context_builder = _attention_and_context(cfg)
    decisions = []
    evaluate = attention.evaluate
    attention.evaluate = lambda e: decisions.append((e.transcript, evaluate(e))) or decisions[-1][1]
    bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
    await bus.start()
    rec = CountingRecorder()
    transcripts = []
    pending: set = set()

    def on_transcript(t) -> None:
        transcripts.append(t)
        task = asyncio.create_task(tm.handle_transcript(t))
        pending.add(task)
        task.add_done_callback(pending.discard)

    segmenter = SpeechSegmenter(config=cfg.vad, sample_rate=cfg.audio.sample_rate, chunk_samples=cfg.audio.chunk_samples)
    frontend = AudioFrontend(mic, vad, segmenter, asr, on_transcript=on_transcript, on_barge_in=None,
                             barge_in=cfg.barge_in, playback_tail_ms=cfg.playback.tail_ms)
    tm = TurnManager(cfg=cfg, bus=bus, attention=attention, context_builder=context_builder, model=model, tts=tts,
                     player=player, memory=memory, frontend=frontend, recorder=rec, session_id="e2e")
    while mic.buffer.get_nowait() is not None:  # audio captured while models loaded is stale
        pass
    fe_task = asyncio.create_task(frontend.run())

    async def utterance(audio: np.ndarray | None, label: str):
        """Feed one utterance (or wait for the operator) and wait for its transcript and finished turn."""
        n_t, n_r = len(transcripts), rec.count
        if audio is None:
            print(f"\n>>> Say now, into the Mac microphone: \"{FIXTURES[FIRST]}\"  (waiting up to 30 s)", flush=True)
            got = await _until(lambda: len(transcripts) > n_t, 30)
        else:
            await play_into(mic.buffer, audio)
            got = await _until(lambda: len(transcripts) > n_t, 30)
        if not check(f"{label}: transcript produced", got,
                     repr(transcripts[-1].text) if got else "no transcript within 30 s"):
            return None
        return transcripts[-1], n_r

    try:
        # Turn 1: ASR, attention, memory, reasoning, TTS, playback.
        row: dict = {}
        got = await utterance(None if args.mic else load_wav(str(wav1)), "turn 1")
        if args.mic:
            mic.suppressed = True  # later turns come from fixtures, put straight into the same buffer
        if got:
            t1, n_r = got
            words = [w.strip().lower() for w in args.expect.split(",") if w.strip()]
            missing = [w for w in words if w not in t1.text.lower()]
            check("ASR transcript contains expected keywords", not missing,
                  f"expected {words}, missing {missing}" if missing else f"{words} in {t1.text!r}")
            dec = decisions[-1][1] if decisions else None
            check("attention decides to reason for an addressed request",
                  dec is not None and dec.should_reason,
                  f"salience={dec.salience:.2f} tier={dec.tier} reason={dec.reason!r}" if dec else "no decision")
            done = await _until(lambda: rec.count > n_r, TURN_TIMEOUT_S)
            row = rec.turns[-1] if done else {}
            tid1 = row.get("turn_id")
            ctx = " ".join(m["content"] for m in model.calls[-1]).lower() if model.calls else ""
            check("memory: seeded fact retrieved into the reasoning context", SEED_KEYWORD in ctx,
                  f"{SEED_FACT!r} {'found' if SEED_KEYWORD in ctx else 'absent'} in {len(model.calls[-1]) if model.calls else 0} messages")
            n_d = model.deltas[-1] if model.deltas else 0
            check("local reasoning streamed text (more than one chunk)", n_d > 1,
                  f"{n_d} deltas from model={model.model_name} url={r.ollama_url}; stats={model.last_stats}")
            n_audio = sum(1 for t, _ in feeds if t == tid1)
            check("TTS produced audio delivered to the player", n_audio > 0 and tid1 in first_audio,
                  f"{n_audio} chunks accepted, outcome={row.get('outcome')}, "
                  f"first_audio={row.get('first_audio')} ms after speech end, "
                  f"player={'StreamingPlayer (speaker)' if args.speaker else 'null sink (FakePlayer)'}")

        # Turn 2: long answer, barge in after first audio. Turn 3 must be clean.
        known = set(first_audio)
        got = await utterance(load_wav(str(DATA / LONG)), "turn 2 (long answer)")
        if got:
            _, n_r = got
            heard = await _until(lambda: bool(set(first_audio) - known), TURN_TIMEOUT_S)
            tid2 = next(iter(set(first_audio) - known)) if heard else None
            await asyncio.sleep(0.3)
            cancel_t = time.time()
            tm.on_barge_in(cancel_t)  # what AudioFrontend calls when barge-in fires
            done = await _until(lambda: rec.count > n_r, 10)
            row2 = rec.turns[-1] if done else {}
            check("barge-in cancels the long answer after first audio",
                  heard and row2.get("outcome") == "cancelled",
                  f"heard={heard} outcome={row2.get('outcome')} turn={str(tid2)[:12]}")
            got = await utterance(load_wav(str(wav1)), "turn 3 (after cancel)")
            if got:
                _, n_r = got
                done = await _until(lambda: rec.count > n_r, TURN_TIMEOUT_S)
                row3 = rec.turns[-1] if done else {}
                late = [t for tid, t in feeds if tid == tid2 and t > cancel_t]
                stale = getattr(player, "stale_played", 0)
                tail = player.last_block.get(tid2, 0) - cancel_t if hasattr(player, "last_block") else None
                check("third request processed after the cancel", row3.get("outcome") == "completed",
                      f"outcome={row3.get('outcome')}")
                check("no stale audio from the cancelled turn after cancel", not late and stale == 0
                      and (tail is None or tail <= 0.05),
                      f"chunks accepted after cancel={len(late)}, stale blocks played={stale}, "
                      f"last cancelled block {'n/a' if tail is None else f'{tail * 1000:.0f} ms'} after cancel")
    finally:
        fe_task.cancel()
        for t in list(pending):
            t.cancel()
        await asyncio.gather(fe_task, *pending, return_exceptions=True)
        await tm.shutdown()
        player.close()
        tts.close()
        asr.close()
        model.close()
        await bus.shutdown()
        memory.close()
        tmp.cleanup()
        if args.mic:
            mic.stop()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wav", help="16-bit PCM WAV of the first request (default: the TTS fixture)")
    p.add_argument("--expect", default="capital,france", help="comma-separated keywords the transcript must contain")
    p.add_argument("--mic", action="store_true", help="first utterance from the real microphone")
    p.add_argument("--speaker", action="store_true", help="play through the real StreamingPlayer (Mac speaker)")
    p.add_argument("--make-fixture", action="store_true", help="synthesize the WAV fixtures with the local TTS")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"  # before any model import: loading must not touch the network
    net = ConnectionRecorder()
    net.install()
    held: dict[str, float] = {}
    stop = threading.Event()
    watcher = watch_remote_tcp(os.getpid(), held, stop)
    check = Checks()
    try:
        if args.make_fixture:
            asyncio.run(make_fixtures())
        else:
            asyncio.run(run(args, check, net))
    except Exception as exc:
        check("run completed without an exception", False, f"{type(exc).__name__}: {exc}")
    finally:
        net.uninstall()
    dests = sorted({str(a) for _, a in net.destinations})
    check("no remote inference: every Python socket connect was loopback", not net.remote,
          f"destinations {dests}" + (f"; REMOTE {net.remote}" if net.remote else ""))
    stop.set()
    watcher.join(timeout=5)
    check("no non-loopback TCP from this process during the run (lsof every 0.5 s, covers native code)", not held,
          ", ".join(f"{c} at {t} s" for c, t in held.items()) or "none")
    if not args.mic or not args.speaker:
        missing = [n for n, used in (("microphone", args.mic), ("speaker", args.speaker)) if not used]
        print(f"NOTE  {' and '.join(missing)} hardware {'was' if len(missing) == 1 else 'were'} NOT exercised"
              f" (use {' '.join('--' + ('mic' if n == 'microphone' else n) for n in missing)})")
    failed = [n for n, ok, _ in check.results if not ok]
    print(f"\nRESULT: {'PASS' if check.ok else 'FAIL'} ({len(check.results) - len(failed)}/{len(check.results)} checks)")
    return 0 if check.ok else 1


if __name__ == "__main__":
    sys.exit(main())
