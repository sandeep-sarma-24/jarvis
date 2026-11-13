#!/usr/bin/env python3
"""Local full-stack benchmark: real models, one Mac, components under concurrent load.

Scenarios (default A-E; F only when selected):
  A  Gemma alone                       (LLM stream)
  B  ASR + Gemma                       sequential: ASR -> LLM;   together: LLM streams while ASR transcribes the next clip
  C  Gemma + TTS                       sequential: LLM -> TTS;   together: TTS synthesizes sentences while the LLM streams
  D  ASR + Gemma + TTS                 both of the above
  E  full system: real TurnManager, attention, persistent SQLite memory (temp db), Granite ASR,
     Gemma, Phonon TTS, null-sink player (scripts/e2e_fakes.FakePlayer, real-time consumption)
  F  E plus one on-demand local vision analysis with the same Ollama model

Spoken input is synthesized by the local TTS (tests/data/*.wav fixtures are used when present),
so no microphone is used, and audio goes to a null sink, so no speaker is used.
Every number printed is measured in this run; anything not measured prints NOT RUN.
Live GPU utilisation needs `sudo powermetrics` and is NOT collected; the GPU share of the
model comes from Ollama's /api/ps size_vram.

Usage:
  python -m scripts.benchmark_local_stack [--scenario A|B|C|D|E|F|all|ABE] [--turns 5]
         [--sustain-seconds 300] [--camera] [--image PATH] [--json out.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from app.config import JarvisConfig, is_loopback, load_config
from app.contracts import percentiles

SCENARIOS = "ABCDEF"
DEFAULT_SCENARIOS = "ABCDE"
DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
NOT_RUN = "NOT RUN"
PROMPTS = [  # (spoken prompt, fixture file or None)
    ("Jarvis, what is the capital of France?", "tts_capital_of_france_16k.wav"),
    ("Jarvis, what is the build server called?", "tts_build_server_16k.wav"),
    ("Jarvis, give me two tips for staying focused.", "tts_focus_tips_16k.wav"),
]
LONG_PROMPT = "Jarvis, explain in detail how a refrigerator keeps food cold, step by step."
VISION_PROMPT = "Jarvis, what do you see on the camera right now?"
SEED_FACT = "The build server is called Atlas and runs in the basement."
GPU_NOTE = ("GPU: live Metal utilisation needs `sudo powermetrics` and was NOT collected; "
            "gpu_fraction below is Ollama's size_vram / size (share of the model in GPU memory).")


# ── Pure parsers (unit-tested) ──────────────────────────────────────────

def select_scenarios(arg: str | None) -> str:
    """'all' -> ABCDEF, None -> ABCDE, else the given letters ('B', 'a,c', 'ABE') in canonical order."""
    if not arg:
        return DEFAULT_SCENARIOS
    if arg.strip().lower() == "all":
        return SCENARIOS
    letters = set(re.sub(r"[\s,]", "", arg.upper()))
    bad = letters - set(SCENARIOS)
    if bad or not letters:
        raise ValueError(f"unknown scenario(s) {sorted(bad) or arg!r}; choose from {SCENARIOS} or all")
    return "".join(s for s in SCENARIOS if s in letters)


def _mb(value: str, unit: str) -> float:
    return float(value) * {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[unit.upper()]


def parse_swapusage(text: str) -> dict[str, float] | None:
    """`sysctl -n vm.swapusage` -> {'total_mb','used_mb','free_mb'}."""
    out = {f"{k}_mb": _mb(v, u) for k, v, u in re.findall(r"(total|used|free)\s*=\s*([\d.]+)([KMG])", text)}
    return out if len(out) == 3 else None


def parse_memory_pressure(text: str) -> int | None:
    """`memory_pressure` output -> system-wide memory free percentage."""
    m = re.search(r"free percentage:\s*(\d+)%", text)
    return int(m.group(1)) if m else None


def parse_thermal(text: str) -> str:
    """`pmset -g therm` -> 'nominal' or the recorded warning lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    flagged = [ln for ln in lines if "No " not in ln and ("level" in ln.lower() or "limit" in ln.lower())]
    if flagged:
        return "; ".join(flagged)
    return "nominal" if any("No thermal warning" in ln for ln in lines) else ("unknown: " + "; ".join(lines))


def parse_footprint(text: str) -> float | None:
    """`footprint PID...` -> summed physical footprint in MB (counts Metal/GPU buffers, unlike RSS)."""
    found = re.findall(r"\]:.*?Footprint:\s*([\d.]+)\s*([KMG])B", text)  # per-process lines, not the Summary
    return round(sum(_mb(v, u) for v, u in found), 1) if found else None


def parse_ollama_ps(data: dict) -> list[dict]:
    """GET /api/ps JSON -> per-model residency (bytes) and GPU share."""
    out = []
    for m in data.get("models") or []:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        out.append({"name": m.get("name") or m.get("model"), "size_mb": round(size / 2**20, 1),
                    "size_vram_mb": round(vram / 2**20, 1),
                    "gpu_fraction": round(vram / size, 3) if size else None,
                    "context_length": m.get("context_length")})
    return out


# ── System probes ──────────────────────────────────────────────────────

def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def system_snapshot() -> dict:
    return {"memory_free_pct": parse_memory_pressure(_run("memory_pressure")),
            "swap": parse_swapusage(_run("sysctl", "-n", "vm.swapusage")),
            "thermal": parse_thermal(_run("pmset", "-g", "therm"))}


def ollama_ps(url: str) -> list[dict] | str:
    try:
        return parse_ollama_ps(httpx.get(f"{url}/api/ps", timeout=3).json())
    except Exception as exc:
        return f"unavailable: {exc}"


class ResourceSampler:
    """Background thread: RSS of this process and of all ollama processes, CPU, memory free %, swap."""

    def __init__(self, interval_s: float = 0.5) -> None:
        import psutil

        self._psutil = psutil
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="sampler", daemon=True)
        self.samples: list[dict] = []

    def _ollama(self) -> tuple[float, list[int]]:
        total, pids = 0, []
        for p in self._psutil.process_iter(["name", "memory_info"]):
            # "llama" also matches llama-server, the runner Ollama spawns for GGUF models.
            if "llama" in (p.info["name"] or "").lower() and p.info["memory_info"] is not None:
                total += p.info["memory_info"].rss
                pids.append(p.pid)
        return total / 2**20, pids

    def _sample(self) -> None:
        swap = parse_swapusage(_run("sysctl", "-n", "vm.swapusage"))
        ollama_rss, pids = self._ollama()
        self.samples.append({
            "jarvis_rss_mb": self._me.memory_info().rss / 2**20, "ollama_rss_mb": ollama_rss,
            # MLX weights live in Metal buffers that RSS does not count; footprint does (no sudo needed).
            "jarvis_footprint_mb": parse_footprint(_run("footprint", str(self._me.pid))),
            "ollama_footprint_mb": parse_footprint(_run("footprint", *map(str, pids))) if pids else None,
            "proc_cpu_pct": self._me.cpu_percent(None), "sys_cpu_pct": self._psutil.cpu_percent(None),
            "memory_free_pct": parse_memory_pressure(_run("memory_pressure")),
            "swap_used_mb": swap["used_mb"] if swap else None,
        })

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def __enter__(self) -> "ResourceSampler":
        self._me = self._psutil.Process()
        self._me.cpu_percent(None)  # first call primes the CPU counters
        self._psutil.cpu_percent(None)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sample()  # a run shorter than one interval still gets a sample

    def summary(self) -> dict:
        def col(k):
            return [s[k] for s in self.samples if s[k] is not None]
        if not self.samples:
            return {"samples": 0}
        out: dict = {"samples": len(self.samples)}
        for k in ("jarvis_rss_mb", "ollama_rss_mb", "jarvis_footprint_mb", "ollama_footprint_mb", "swap_used_mb"):
            out[f"peak_{k}"] = round(max(col(k)), 1) if col(k) else None
        out["min_memory_free_pct"] = min(col("memory_free_pct")) if col("memory_free_pct") else None
        for k in ("proc_cpu_pct", "sys_cpu_pct"):
            v = col(k)
            out[f"mean_{k}"], out[f"max_{k}"] = round(sum(v) / len(v), 1), round(max(v), 1)
        return out


# ── Components ─────────────────────────────────────────────────────────

def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    n = int(len(audio) * 16000 / rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


async def synth_clip_16k(tts, text: str) -> np.ndarray:
    """Speech for `text` from the local TTS, as 16 kHz mono float32 (ASR input)."""
    r = await asyncio.to_thread(tts.synthesize, text)
    if not r.success:
        raise RuntimeError(f"TTS failed to synthesize {text!r}: {r.error}")
    return to_16k(r.audio, r.sample_rate)


class Stack:
    """Loads each component once, on first use, and records its load time."""

    def __init__(self, cfg: JarvisConfig) -> None:
        self.cfg = cfg
        self.loads_ms: dict[str, float] = {}
        self.model = self.asr = self.tts = None
        self.clips: dict[str, np.ndarray] = {}

    async def llm(self):
        if self.model is None:
            from app.reasoning import build_reasoning_model

            self.model = build_reasoning_model(self.cfg.reasoning)
            before = ollama_ps(self.cfg.reasoning.ollama_url)
            t0 = time.perf_counter()
            await asyncio.to_thread(self.model.warm_up)
            self.loads_ms["llm_warm_up"] = round((time.perf_counter() - t0) * 1000, 1)
            self.loads_ms["llm_was_resident_before"] = bool(
                isinstance(before, list) and any(m["name"] == self.cfg.reasoning.model for m in before))
        return self.model

    async def get_asr(self):
        if self.asr is None:
            from app.audio.asr import SpeechRecognizer

            self.asr = SpeechRecognizer(self.cfg.asr)
            t0 = time.perf_counter()
            await asyncio.to_thread(self.asr.load)
            self.loads_ms["asr_load"] = round((time.perf_counter() - t0) * 1000, 1)
        return self.asr

    async def get_tts(self):
        if self.tts is None:
            from app.tts.local_tts import PhononTTS

            t = self.cfg.tts
            self.tts = PhononTTS(model_id=t.model, voice=t.voice, quant=t.quant)
            t0 = time.perf_counter()
            await asyncio.to_thread(self.tts.load)
            self.loads_ms["tts_load"] = round((time.perf_counter() - t0) * 1000, 1)
        return self.tts

    async def clip(self, text: str, fixture: str | None = None) -> np.ndarray:
        if text not in self.clips:
            path = DATA / fixture if fixture else None
            if path is not None and path.exists():
                from scripts.benchmark_audio import load_wav

                self.clips[text] = load_wav(str(path))
            else:
                self.clips[text] = await synth_clip_16k(await self.get_tts(), text)
        return self.clips[text]

    def resident(self) -> list[str]:
        return [n for n, c in (("asr", self.asr), ("tts", self.tts)) if c is not None]

    def close(self) -> None:
        for c in (self.asr, self.tts, self.model):
            if c is not None:
                c.close()


def _messages(text: str) -> list[dict[str, str]]:
    from app.context.builder import _load_system_prompt

    return [{"role": "system", "content": _load_system_prompt()}, {"role": "user", "content": text}]


async def one_turn(stack: Stack, prompt: str, *, clip: np.ndarray | None, use_tts: bool,
                   together: bool, side_clip: np.ndarray | None = None) -> dict:
    """One request. t0 = end of speech (or request time when there is no speech input).

    together: TTS speaks sentences while the LLM still streams, and side_clip (if any) is
    transcribed by ASR while the LLM streams. Otherwise every stage runs to completion first.
    """
    from app.orchestration.sentences import SentenceSplitter

    cfg = stack.cfg.reasoning
    model = await stack.llm()
    tts = await stack.get_tts() if use_tts else None
    loop = asyncio.get_running_loop()
    row: dict = {}
    t0 = time.perf_counter()
    ms = lambda: round((time.perf_counter() - t0) * 1000, 1)  # noqa: E731
    text = prompt
    if clip is not None:
        r = await (await stack.get_asr()).transcribe_async(clip, 16000)
        row["asr_ms"], row["transcript"] = round(r.latency_ms, 1), r.text
        text = r.text or prompt

    deltas: asyncio.Queue = asyncio.Queue()

    def produce() -> None:
        try:
            for d in model.stream(_messages(text), max_tokens=cfg.max_output_tokens, temperature=cfg.temperature):
                loop.call_soon_threadsafe(deltas.put_nowait, d)
        except Exception as exc:
            loop.call_soon_threadsafe(deltas.put_nowait, exc)
        loop.call_soon_threadsafe(deltas.put_nowait, None)

    chunks = 0

    def sink(_chunk) -> bool:
        nonlocal chunks
        chunks += 1
        row.setdefault("first_audible_ms", ms())
        return True

    async def speak(q: asyncio.Queue) -> None:
        while (s := await q.get()) is not None:
            st = await tts.stream_to(s, sink)
            row.setdefault("tts_first_chunk_ms", st.first_chunk_ms and round(st.first_chunk_ms, 1))

    side = None
    if together and side_clip is not None:
        async def side_asr():
            s0 = ms()
            r = await (await stack.get_asr()).transcribe_async(side_clip, 16000)
            return s0, ms(), round(r.latency_ms, 1)
        side = asyncio.create_task(side_asr())
    sentences: asyncio.Queue = asyncio.Queue()
    speaker = asyncio.create_task(speak(sentences)) if tts is not None and together else None
    splitter = SentenceSplitter()
    reply, error = "", None
    producer = loop.run_in_executor(None, produce)
    while (d := await deltas.get()) is not None:
        if isinstance(d, Exception):
            error = f"{type(d).__name__}: {d}"
            continue
        row.setdefault("llm_first_delta_ms", ms())
        reply += d
        if speaker is not None:
            for s in splitter.feed(d):
                sentences.put_nowait(s)
    await producer
    row["llm_done_ms"] = ms()
    stats = dict(model.last_stats)
    row["ttft_ms"], row["tokens_per_second"] = stats.get("ttft_ms"), stats.get("tokens_per_second")
    row["output_tokens"] = stats.get("output_tokens")
    if speaker is not None:
        if tail := splitter.flush():
            sentences.put_nowait(tail)
        sentences.put_nowait(None)
        await speaker
    elif tts is not None and reply.strip():
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(reply.strip())
        q.put_nowait(None)
        await speak(q)
    if side is not None:
        s0, s1, lat = await side
        row["side_asr_ms"] = lat
        # Overlap is real only if ASR ran while the LLM had not finished streaming.
        row["side_asr_overlapped_llm"] = s0 < row["llm_done_ms"] and s1 > row.get("llm_first_delta_ms", 1e18)
    row["e2e_ms"], row["audio_chunks"] = ms(), chunks if tts is not None else None
    row["reply_chars"] = len(reply)
    if error or not reply.strip():
        row["error"] = error or "empty reply"
    return row


SCENARIO_PARTS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}


async def run_component_scenario(stack: Stack, name: str, turns: int) -> dict:
    use_asr, use_tts = SCENARIO_PARTS[name]
    clips = [await stack.clip(p, f) for p, f in PROMPTS] if use_asr else [None] * len(PROMPTS)
    modes = {"sequential": False} if name == "A" else {"sequential": False, "together": True}
    out: dict = {}
    for mode, together in modes.items():
        rows = []
        for i in range(turns):
            k = i % len(PROMPTS)
            try:
                rows.append(await one_turn(stack, PROMPTS[k][0], clip=clips[k], use_tts=use_tts, together=together,
                                           side_clip=clips[(k + 1) % len(PROMPTS)] if use_asr else None))
            except Exception as exc:
                rows.append({"error": f"{type(exc).__name__}: {exc}"})
            await asyncio.sleep(0.2)
        out[mode] = rows
    return out


async def run_full_system(stack: Stack, turns: int, *, vision: bool, camera: bool, image: str | None) -> dict:
    """Scenario E/F: the real TurnManager with memory in a temp SQLite file."""
    from app.contracts import Transcript
    from app.events.bus import EventBus
    from app.memory.manager import MemoryManager
    from app.memory.sqlite_store import SQLiteStore
    from app.memory.types import MemoryType
    from app.orchestration.turns import TurnManager
    from scripts.benchmark_e2e import _attention_and_context, _wire_player, drive_turns
    from scripts.e2e_fakes import CountingRecorder, FakePlayer

    cfg = stack.cfg
    cfg.perception.enabled = vision
    model, asr, tts = await stack.llm(), await stack.get_asr(), await stack.get_tts()
    prompts = PROMPTS + ([(VISION_PROMPT, None)] if vision else [])
    clips = [await stack.clip(p, f) for p, f in prompts]
    out: dict = {"llm_stats": []}
    vis = None
    if vision:
        vis = await vision_setup(cfg, camera, image)
        out["vision_standalone"] = await asyncio.to_thread(vision_once, vis, cfg)

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "jarvis.db"
        cfg.memory.database_path = str(db)
        memory = MemoryManager(SQLiteStore(db), cross_session=cfg.memory.cross_session)
        memory.start_session()
        memory.remember(SEED_FACT, MemoryType.FACT, importance=0.8)
        recorder = CountingRecorder()
        bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
        await bus.start()
        player = FakePlayer(volume=cfg.playback.volume, max_queue_seconds=cfg.playback.max_queue_seconds)
        attention, context_builder = _attention_and_context(cfg)
        tm = TurnManager(cfg=cfg, bus=bus, attention=attention, context_builder=context_builder, model=model,
                         tts=tts, player=player, memory=memory, recorder=recorder, session_id="bench",
                         on_demand_vision=(vis["camera"], vis["analyzer"]) if vis else None)
        _wire_player(tm, player)
        player.start(tts.sample_rate)
        order = list(range(len(PROMPTS)))
        seq = [order[i % len(order)] for i in range(turns)] + ([len(PROMPTS)] if vision else [])

        async def transcripts():
            for n, k in enumerate(seq):
                if n:
                    out["llm_stats"].append(dict(model.last_stats))
                t_end = time.time()  # the synthetic speaker has just stopped
                r = await asr.transcribe_async(clips[k], 16000)
                yield Transcript(text=r.text, speech_start_time=t_end - len(clips[k]) / 16000, speech_end_time=t_end,
                                 segment_emitted_time=t_end, asr_latency_ms=r.latency_ms,
                                 audio_seconds=len(clips[k]) / 16000)
        try:
            res = await drive_turns(tm, player, recorder, transcripts(), gap_s=0.5)
            out["llm_stats"].append(dict(model.last_stats))
        finally:
            await tm.shutdown()
            player.close()
            await bus.shutdown()
            memory.close()
    rows = list(recorder.turns)
    out.update(rows=rows, sent=res.sent, timeouts=res.timeouts, player_first_audio_ms=res.player_first_audio_ms,
               player_stats=player.stats, stale_chunks_played=player.stale_played)
    if vis:
        out["vision_turn_analyzer_metrics"] = vis["analyzer"].metrics
        out["vision_turn_timeout_s"] = cfg.perception.on_demand_timeout_seconds
        out["vision_image_source"] = vis["source"]
        vis["analyzer"].close()
    return out


async def vision_setup(cfg: JarvisConfig, camera: bool, image: str | None) -> dict:
    """Analyzer on the reasoning model (same model, no second one) plus a frame source."""
    from app.perception.local_vision import LocalVisionAnalyzer, OllamaVisionBackend
    from app.perception.types import Frame

    v = cfg.perception.vision
    backend = OllamaVisionBackend(model=cfg.reasoning.model, base_url=cfg.reasoning.ollama_url,
                                  timeout_seconds=v.timeout_seconds, temperature=v.temperature,
                                  num_ctx=cfg.reasoning.num_ctx, keep_alive=cfg.reasoning.keep_alive)
    analyzer = LocalVisionAnalyzer(backend=backend, prompt=v.prompt, timeout_seconds=v.timeout_seconds,
                                   max_image_size=v.max_image_size)
    frame, source = None, None
    if camera:
        from app.perception.camera import OpenCVCamera

        c = cfg.perception.camera
        cam = OpenCVCamera(device_index=c.device_index, width=c.width, height=c.height)
        try:
            await cam.start()
            frame = await cam.capture()
            source = "camera"
        except Exception as exc:
            source = f"camera failed ({exc}); "
        finally:
            await cam.stop()
    if frame is None:
        jpeg = Path(image).read_bytes() if image else generated_image_jpeg()
        frame = Frame(timestamp=time.time(), data=jpeg, width=0, height=0, format="jpeg")
        source = (source or "") + (f"image {image}" if image else "generated test image")

    class StillCamera:
        async def capture(self):
            return Frame(timestamp=time.time(), data=frame.data, width=frame.width, height=frame.height,
                         format=frame.format)

    return {"camera": StillCamera(), "analyzer": analyzer, "backend": backend, "frame": frame,
            "prompt": analyzer._prompt, "source": source}


def generated_image_jpeg() -> bytes:
    """A 512x384 JPEG with a red square, a blue circle and the text JARVIS TEST."""
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 384), (235, 235, 225))
    d = ImageDraw.Draw(img)
    d.rectangle((40, 60, 200, 220), fill=(200, 30, 30))
    d.ellipse((280, 80, 460, 260), fill=(30, 60, 200))
    d.text((180, 310), "JARVIS TEST", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()



def vision_once(vis: dict, cfg: JarvisConfig) -> dict:
    """One request through the app's backend; if it fails, one plain request without
    `format: json`, to tell "the model rejects images" apart from "the backend's request shape fails"."""
    t0 = time.perf_counter()
    b64 = vis["analyzer"]._preprocess(vis["frame"])
    out: dict = {"image_jpeg_bytes": len(vis["frame"].data)}
    try:
        raw = vis["backend"].infer(b64, vis["prompt"])
        parsed = vis["analyzer"]._parse(raw, time.time())
        out.update(app_backend_ok=True, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                   parsed=parsed is not None, description=(parsed.description if parsed else raw)[:200])
        return out
    except Exception as exc:
        detail = getattr(getattr(exc, "response", None), "text", "") or ""
        out.update(app_backend_ok=False, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                   app_backend_error=f"{type(exc).__name__}: {str(exc).splitlines()[0]} {detail[:200]}".strip())
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{cfg.reasoning.ollama_url}/api/chat", timeout=cfg.perception.vision.timeout_seconds, json={
            "model": cfg.reasoning.model, "stream": False, "think": False, "keep_alive": cfg.reasoning.keep_alive,
            "messages": [{"role": "user", "content": "Describe this image in one sentence.", "images": [b64]}],
            "options": {"num_ctx": cfg.reasoning.num_ctx, "num_predict": 80}})
        ok = r.status_code < 400
        out.update(plain_image_request_ok=ok, plain_latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                   plain_reply=(r.json().get("message", {}).get("content", "") if ok else r.text)[:200])
    except Exception as exc:
        out.update(plain_image_request_ok=False, plain_reply=f"{type(exc).__name__}: {exc}")
    return out


async def sustain(stack: Stack, seconds: float) -> dict:
    """Back-to-back long LLM turns for `seconds`; tokens/s of the first vs last window, thermal state."""
    model = await stack.llm()
    cfg = stack.cfg.reasoning
    window = min(60.0, seconds / 2)
    rows, thermal = [], [(0.0, parse_thermal(_run("pmset", "-g", "therm")))]
    start = time.monotonic()
    while (t := time.monotonic() - start) < seconds:
        try:
            await asyncio.to_thread(lambda: list(model.stream(_messages(LONG_PROMPT), max_tokens=cfg.max_output_tokens,
                                                              temperature=cfg.temperature)))
            rows.append((t, time.monotonic() - start, model.last_stats.get("tokens_per_second")))
        except Exception as exc:
            rows.append((t, time.monotonic() - start, None))
            print(f"  sustain turn failed: {exc}")
        state = parse_thermal(_run("pmset", "-g", "therm"))
        if state != thermal[-1][1]:
            thermal.append((round(time.monotonic() - start, 1), state))
    elapsed = time.monotonic() - start
    first = [r[2] for r in rows if r[0] < window and r[2]]
    last = [r[2] for r in rows if r[1] > elapsed - window and r[2]]
    return {"seconds": round(elapsed, 1), "turns": len(rows), "window_s": window,
            "failures": sum(r[2] is None for r in rows),
            "tps_first_window": percentiles(first), "tps_last_window": percentiles(last),
            "thermal_changes": thermal}


# ── Reporting ──────────────────────────────────────────────────────────

def fmt(stats: dict | None, unit: str = "ms") -> str:
    if not stats or not stats.get("n"):
        return NOT_RUN
    return f"n={stats['n']} p50={stats['p50']:.1f} p95={stats['p95']:.1f} p99={stats['p99']:.1f} {unit}"


def col(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]


def summarize_rows(rows: list[dict]) -> dict:
    keys = ("asr_ms", "ttft_ms", "tokens_per_second", "llm_done_ms", "tts_first_chunk_ms", "first_audible_ms",
            "e2e_ms", "side_asr_ms")
    s = {k: percentiles(col(rows, k)) for k in keys}
    s["failures"] = [r["error"] for r in rows if r.get("error")]
    s["turns"] = len(rows)
    ov = [r["side_asr_overlapped_llm"] for r in rows if "side_asr_overlapped_llm" in r]
    s["side_asr_overlap_confirmed"] = f"{sum(ov)}/{len(ov)}" if ov else None
    return s


LABELS = {"asr_ms": "ASR latency", "ttft_ms": "LLM TTFT (Ollama, from request)",
          "tokens_per_second": "LLM tokens/s", "llm_done_ms": "LLM done (from t0)",
          "tts_first_chunk_ms": "TTS first chunk (from its call)", "first_audible_ms": "first audible (t0 -> 1st chunk)",
          "e2e_ms": "end to end", "side_asr_ms": "concurrent ASR latency"}


def print_rows_summary(s: dict) -> None:
    for k, label in LABELS.items():
        print(f"    {label:34s} {fmt(s[k], 'tok/s' if k == 'tokens_per_second' else 'ms')}")
    if s.get("side_asr_overlap_confirmed"):
        print(f"    {'ASR overlapped LLM streaming':34s} {s['side_asr_overlap_confirmed']} turns")
    print(f"    {'failures':34s} {len(s['failures'])} {s['failures'][:3] if s['failures'] else ''}")


def summarize_full(r: dict) -> dict:
    from app.contracts import LatencyRecorder

    rec = LatencyRecorder(maxlen=max(1, len(r["rows"])))
    rec.turns.extend(r["rows"])
    stats = r["llm_stats"]
    return {"stages_ms_since_speech_end": rec.summary(),
            "player_first_audio_ms": percentiles(r["player_first_audio_ms"]),
            "ttft_ms": percentiles(col(stats, "ttft_ms")), "tokens_per_second": percentiles(col(stats, "tokens_per_second")),
            "outcomes": {o: [x["outcome"] for x in r["rows"]].count(o) for o in {x["outcome"] for x in r["rows"]}},
            "timeouts": r["timeouts"], "max_playback_queue_s": r["player_stats"]["max_queue_seconds"],
            "chunks_dropped_stale": r["player_stats"]["chunks_dropped_stale"],
            "stale_chunks_played": r["stale_chunks_played"]}


def print_full(s: dict) -> None:
    for stage in ("asr_done", "context_ready", "llm_first_token", "first_sentence", "tts_first_chunk", "first_audio",
                  "llm_done", "playback_done"):
        print(f"    {stage + ' (ms since speech end)':34s} {fmt(s['stages_ms_since_speech_end'].get(stage))}")
    print(f"    {'first audible (player)':34s} {fmt(s['player_first_audio_ms'])}")
    print(f"    {'LLM TTFT (Ollama)':34s} {fmt(s['ttft_ms'])}")
    print(f"    {'LLM tokens/s':34s} {fmt(s['tokens_per_second'], 'tok/s')}")
    print(f"    outcomes {s['outcomes']} timeouts {s['timeouts']}")
    print(f"    max playback queue {s['max_playback_queue_s']:.2f} s; dropped stale chunks "
          f"{s['chunks_dropped_stale']:.0f}; stale chunks played {s['stale_chunks_played']} (must be 0)")


def print_resources(before: dict, after: dict, res: dict, ps) -> None:
    def swap(s):
        return f"{s['used_mb']:.0f} MB" if s else NOT_RUN
    def v(x, unit=""):
        return NOT_RUN if x is None else f"{x}{unit}"
    print(f"    RSS peak: jarvis {v(res.get('peak_jarvis_rss_mb'), ' MB')}, ollama {v(res.get('peak_ollama_rss_mb'), ' MB')} "
          f"({res.get('samples', 0)} samples)")
    print(f"    physical footprint peak (incl. Metal buffers): jarvis {v(res.get('peak_jarvis_footprint_mb'), ' MB')}, "
          f"ollama {v(res.get('peak_ollama_footprint_mb'), ' MB')}")
    print(f"    CPU: process mean {v(res.get('mean_proc_cpu_pct'), '%')} max {v(res.get('max_proc_cpu_pct'), '%')}; "
          f"system mean {v(res.get('mean_sys_cpu_pct'), '%')} max {v(res.get('max_sys_cpu_pct'), '%')}")
    print(f"    memory free: before {v(before['memory_free_pct'], '%')}, min during {v(res.get('min_memory_free_pct'), '%')}, "
          f"after {v(after['memory_free_pct'], '%')}")
    print(f"    swap used: before {swap(before['swap'])}, peak {v(res.get('peak_swap_used_mb'), ' MB')}, after {swap(after['swap'])}")
    print(f"    thermal: before {before['thermal']}, after {after['thermal']}")
    print(f"    ollama /api/ps: {ps}")


async def run(args) -> dict:
    cfg = load_config()
    url = cfg.reasoning.ollama_url
    if not is_loopback(url):
        raise SystemExit(f"reasoning.ollama_url {url} is not loopback")
    cfg.memory.enabled = True
    cfg.perception.enabled = False
    stack = Stack(cfg)
    report: dict = {"model": cfg.reasoning.model, "ollama_url": url, "turns": args.turns,
                    "scenarios": {}, "gpu_note": GPU_NOTE}
    print(f"Local stack benchmark: model {cfg.reasoning.model} at {url}; scenarios {args.scenarios}; "
          f"turns {args.turns} per mode")
    print(GPU_NOTE)
    try:
        for name in args.scenarios:
            print(f"\n=== Scenario {name} ===")
            before = system_snapshot()
            with ResourceSampler() as sampler:
                if name in SCENARIO_PARTS:
                    raw = await run_component_scenario(stack, name, args.turns)
                    summary = {mode: summarize_rows(rows) for mode, rows in raw.items()}
                    for mode, s in summary.items():
                        print(f"  [{mode}]")
                        print_rows_summary(s)
                else:
                    raw = await run_full_system(stack, args.turns, vision=name == "F", camera=args.camera,
                                                image=args.image)
                    summary = summarize_full(raw)
                    print("  [full pipeline: LLM and TTS overlap inside TurnManager]")
                    print_full(summary)
                    if name == "F":
                        print(f"    vision image: {raw['vision_image_source']}")
                        print(f"    vision standalone: {raw['vision_standalone']}")
                        print(f"    vision turn analyzer: {raw['vision_turn_analyzer_metrics']} "
                              f"(turn waits at most {raw['vision_turn_timeout_s']} s)")
                if name not in "EF":
                    print(f"    {'dropped chunks / playback queue':34s} {NOT_RUN} (null sink, no player in A-D)")
            after = system_snapshot()
            ps = ollama_ps(url)
            print(f"  resources (in-process models: {stack.resident() or 'none'}):")
            print_resources(before, after, sampler.summary(), ps)
            report["scenarios"][name] = {"summary": summary, "raw": raw, "system_before": before, "system_after": after,
                                         "resources": sampler.summary(), "ollama_ps": ps,
                                         "in_process_models": stack.resident()}
        if args.sustain_seconds:
            print(f"\n=== Sustained generation, {args.sustain_seconds:.0f} s ===")
            with ResourceSampler(interval_s=2.0) as sampler:
                s = await sustain(stack, args.sustain_seconds)
            s["resources"] = sampler.summary()
            print(f"    turns {s['turns']} in {s['seconds']} s, failures {s['failures']}")
            print(f"    tokens/s first {s['window_s']:.0f} s: {fmt(s['tps_first_window'], 'tok/s')}")
            print(f"    tokens/s last {s['window_s']:.0f} s:  {fmt(s['tps_last_window'], 'tok/s')}")
            print(f"    thermal (t s, state): {s['thermal_changes']}")
            report["sustain"] = s
        else:
            print(f"\nSustained generation: {NOT_RUN} (use --sustain-seconds N)")
        print(f"\nload / warm-up ms (measured once, first use): {stack.loads_ms or NOT_RUN}")
        print("microphone and speaker: NOT exercised (TTS-synthesized input, null-sink output)")
        report["loads_ms"] = stack.loads_ms
    finally:
        stack.close()
    return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default=None, help="A-F letters (e.g. A, ABE) or all; default A-E")
    p.add_argument("--turns", type=int, default=5, help="turns per mode per scenario")
    p.add_argument("--sustain-seconds", type=float, default=0.0)
    p.add_argument("--camera", action="store_true", help="scenario F: try the real camera first")
    p.add_argument("--image", help="scenario F: image file instead of the generated test image")
    p.add_argument("--json", metavar="PATH")
    args = p.parse_args(argv)
    try:
        args.scenarios = select_scenarios(args.scenario)
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached; never download during a benchmark
    report = asyncio.run(run(args))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
