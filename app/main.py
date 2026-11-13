"""JARVIS — Multimodal perception → Fusion → Attention → Context → LLM → Response Decision → TTS.

Entry point and lifecycle wiring. The request path lives in app.orchestration:
AudioFrontend → TurnManager → streaming reasoning → streaming TTS → StreamingPlayer.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import signal
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

from app.config import JarvisConfig, enforce_local_only, load_config
from app.events.bus import EventBus
from app.events.types import Event, SpeechEvent, VisualEvent
from app.attention.engine import AttentionEngine
from app.fusion.engine import MultimodalFusion
from app.memory.working import WorkingMemory

logger = logging.getLogger("jarvis")

_metrics = {
    "visual_events_received": 0,
    "visual_events_attention_worthy": 0,
}


# ── Console subscriber ──────────────────────────────────────────────────

async def _console_handler(event: Event) -> None:
    if not isinstance(event, SpeechEvent):
        return
    ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
    ms = int((event.timestamp % 1) * 1000)
    md = event.metadata
    asr_ms = md.get("asr_latency_ms", "?")
    total_ms = md.get("total_speech_to_event_ms", "?")
    print(
        f"\n[{ts}.{ms:03d}] SPEECH\n"
        f"  event_id={event.id[:12]}…\n"
        f"  duration={event.duration_seconds:.2f}s\n"
        f"  confidence={event.confidence:.2f}\n"
        f"  asr_latency={asr_ms}ms\n"
        f"  speech_to_event={total_ms}ms\n"
        f'\n  "{event.transcript}"\n'
    )


# ── Visual attention handler ──────────────────────────────────────────────

def _build_visual_attention_handler(
    attention: AttentionEngine,
    visual_context_ring: deque,
    max_age_seconds: float,
    fusion: MultimodalFusion | None = None,
):
    """Build a handler that routes VisualEvents through the attention engine.

    Phase 8: ALL visual events enter fusion for temporal context (independent
    of attention). Attention only gates the legacy visual_context_ring.
    No automatic LLM response — attention decides relevance only.
    Entries older than max_age_seconds are pruned on each event.
    """
    async def _handler(event: Event) -> None:
        if not isinstance(event, VisualEvent):
            return

        _metrics["visual_events_received"] += 1

        # Phase 8: visual events enter fusion independently of attention
        if fusion:
            fusion.ingest(event)

        # Prune expired entries: (timestamp, description)
        now = time.time()
        while visual_context_ring and now - visual_context_ring[0][0] > max_age_seconds:
            visual_context_ring.popleft()

        decision = attention.evaluate_visual(event)

        if decision.should_reason:
            _metrics["visual_events_attention_worthy"] += 1
            visual_context_ring.append((now, event.description))

    return _handler


# ── Lifecycle ────────────────────────────────────────────────────────────

async def _until_shutdown(shutdown: asyncio.Event, aw) -> bool:
    """Await `aw` unless shutdown is requested first. True if `aw` finished."""
    fut = asyncio.ensure_future(aw)
    stopper = asyncio.ensure_future(shutdown.wait())
    await asyncio.wait({fut, stopper}, return_when=asyncio.FIRST_COMPLETED)
    stopper.cancel()
    if fut.done():
        fut.result()  # re-raise a failed load or a crashed frontend
        return True
    fut.cancel()
    return False


async def run(cfg: JarvisConfig | None = None) -> None:
    # Hardware and model imports stay inside run() so `import app.main` stays cheap.
    from app.audio.asr import SpeechRecognizer
    from app.audio.capture import MicrophoneCapture
    from app.audio.frontend import AudioFrontend
    from app.audio.segmentation import SpeechSegmenter
    from app.audio.vad import VoiceActivityDetector
    from app.context.builder import ContextBuilder
    from app.contracts import LatencyRecorder
    from app.orchestration.turns import TurnManager

    cfg = cfg or load_config()
    enforce_local_only(cfg)  # a programmatic JarvisConfig skips load_config's check

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Signal handlers first, so Ctrl+C during model loading still shuts down cleanly.
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    session_id = uuid.uuid4().hex
    logger.info("JARVIS session %s starting", session_id[:12])
    logger.info("Microphone capture will be active — audio is processed locally, never stored.")
    r = cfg.reasoning
    logger.info(
        "Runtime policy: local_only=%s; reasoning backend=ollama model=%s url=%s num_ctx=%d keep_alive=%s",
        cfg.runtime.local_only, r.model, r.ollama_url, r.num_ctx, r.keep_alive,
    )

    cleanup: list[tuple[str, object]] = []  # (name, callable), run in reverse in finally
    recorder = LatencyRecorder()
    frontend = None
    attention: AttentionEngine | None = None
    perception_pipeline = None
    fusion: MultimodalFusion | None = None
    bus: EventBus | None = None
    player = None
    pending: set[asyncio.Task] = set()

    try:
        # 1. Memory (Phase 5)
        memory_manager = None
        if cfg.memory.enabled:
            from app.memory.manager import MemoryManager
            from app.memory.sqlite_store import SQLiteStore

            db_path = Path(cfg.memory.database_path).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            memory_manager = MemoryManager(SQLiteStore(db_path), cross_session=cfg.memory.cross_session)
            cleanup.append(("memory", memory_manager.close))
            memory_manager.start_session()
            logger.info(
                "Memory active — db=%s session=%s",
                db_path, memory_manager.conversation.session_id[:12],
            )

        # 2. Event bus, fusion, attention
        bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
        bus.subscribe(SpeechEvent, _console_handler)
        visual_context_ring: deque[tuple[float, str]] = deque(maxlen=cfg.perception.max_visual_events)

        if cfg.fusion.enabled:
            fusion = MultimodalFusion(cfg.fusion)
            logger.info(
                "Fusion active — window=%.1fs max_events=%d max_chars=%d",
                cfg.fusion.window_seconds, cfg.fusion.max_events, cfg.fusion.max_characters,
            )

            async def _fusion_speech_handler(event: Event) -> None:
                if isinstance(event, SpeechEvent):
                    fusion.ingest(event)
            bus.subscribe(SpeechEvent, _fusion_speech_handler)

        if cfg.attention.enabled:
            attention = AttentionEngine(
                assistant_names=cfg.attention.assistant_names,
                reason_threshold=cfg.attention.reason_threshold,
                working_memory=WorkingMemory(max_size=cfg.attention.working_memory_size),
            )
            logger.info("Attention engine active — names=%s threshold=%.2f",
                        cfg.attention.assistant_names, cfg.attention.reason_threshold)

        # 3. Reasoning, TTS and player: constructed here, loaded below
        model = None
        if cfg.reasoning.enabled:
            from app.reasoning import build_reasoning_model

            model = build_reasoning_model(cfg.reasoning)
            cleanup.append(("reasoning", model.close))
            logger.info("Reasoning active — model=%s", model.model_name)

        tts = None
        if cfg.tts.enabled and cfg.playback.enabled:
            from app.tts.local_tts import PhononTTS
            from app.tts.player import StreamingPlayer

            tts = PhononTTS(model_id=cfg.tts.model, voice=cfg.tts.voice, quant=cfg.tts.quant)
            cleanup.append(("tts", tts.close))
            # Callbacks fire on the PortAudio thread; TurnManager hops to the loop.
            player = StreamingPlayer(
                volume=cfg.playback.volume,
                max_queue_seconds=cfg.playback.max_queue_seconds,
                on_first_audio=lambda tid: turns.on_first_audio(tid),
                on_done=lambda tid, outcome: turns.on_playback_done(tid, outcome),
            )
            cleanup.append(("player", player.close))
            logger.info("TTS active — model=%s voice=%s", cfg.tts.model, cfg.tts.voice)

        # 4. Visual perception components (Phase 6)
        camera = analyzer = None
        on_demand_vision = None
        if cfg.perception.enabled and cfg.perception.camera.enabled and attention is not None:
            from app.perception.camera import OpenCVCamera
            from app.perception.visual import MockVisualAnalyzer

            cam_cfg = cfg.perception.camera
            camera = OpenCVCamera(device_index=cam_cfg.device_index, width=cam_cfg.width, height=cam_cfg.height)
            vis_cfg = cfg.perception.vision
            if vis_cfg.enabled:
                if vis_cfg.provider != "ollama":
                    raise ValueError(
                        f"Unsupported vision provider: {vis_cfg.provider!r} (only 'ollama' is supported)"
                    )
                from app.perception.local_vision import LocalVisionAnalyzer, OllamaVisionBackend

                analyzer = LocalVisionAnalyzer(
                    backend=OllamaVisionBackend(
                        model=vis_cfg.model,
                        base_url=vis_cfg.ollama_url,
                        timeout_seconds=vis_cfg.timeout_seconds,
                        temperature=vis_cfg.temperature,
                        num_ctx=cfg.reasoning.num_ctx,
                        keep_alive=cfg.reasoning.keep_alive,
                    ),
                    prompt=vis_cfg.prompt,
                    timeout_seconds=vis_cfg.timeout_seconds,
                    max_image_size=vis_cfg.max_image_size,
                )
                cleanup.append(("vision", analyzer.close))
                on_demand_vision = (camera, analyzer)
                logger.info("Vision analyzer: LocalVisionAnalyzer (model=%s)", vis_cfg.model)
                if cfg.runtime.local_only and vis_cfg.model != cfg.reasoning.model:
                    logger.warning(
                        "Vision model %s differs from reasoning model %s: two models will be resident in Ollama",
                        vis_cfg.model, cfg.reasoning.model,
                    )
            else:
                analyzer = MockVisualAnalyzer()
                logger.info("Vision analyzer: MockVisualAnalyzer")

        # 5. Audio frontend and turn manager
        mic = MicrophoneCapture(cfg.audio)
        vad = VoiceActivityDetector(threshold=cfg.vad.threshold)
        asr = SpeechRecognizer(cfg.asr)
        cleanup.append(("asr", asr.close))
        segmenter = SpeechSegmenter(
            config=cfg.vad, sample_rate=cfg.audio.sample_rate, chunk_samples=cfg.audio.chunk_samples,
        )

        def _on_transcript(transcript) -> None:
            task = asyncio.create_task(turns.handle_transcript(transcript))
            pending.add(task)
            task.add_done_callback(pending.discard)

        frontend = AudioFrontend(
            mic, vad, segmenter, asr,
            on_transcript=_on_transcript,
            on_barge_in=(lambda t: turns.on_barge_in(t)) if cfg.barge_in.enabled else None,
            barge_in=cfg.barge_in,
            playback_tail_ms=cfg.playback.tail_ms,
        )
        turns = TurnManager(
            cfg=cfg, bus=bus, attention=attention,
            context_builder=ContextBuilder(
                max_messages=cfg.reasoning.max_context_messages,
                max_characters=cfg.reasoning.max_context_characters,
                max_memory_items=cfg.memory.max_retrieved_memories,
                max_visual_events=cfg.perception.max_visual_events,
                max_visual_context_characters=cfg.perception.max_visual_context_characters,
            ),
            model=model, tts=tts, player=player, memory=memory_manager, fusion=fusion,
            frontend=frontend, recorder=recorder, on_demand_vision=on_demand_vision,
            session_id=session_id, visual_context_ring=visual_context_ring,
        )
        # Shut down before the player, TTS and model it drives are closed.
        cleanup.append(("turns", turns.shutdown))

        if camera is not None:
            from app.perception.pipeline import PerceptionPipeline
            from app.perception.sampler import FrameSampler
            from app.perception.visual import ChangeDetector, EventDeduplicator

            cam_cfg = cfg.perception.camera
            perception_pipeline = PerceptionPipeline(
                camera=camera,
                sampler=FrameSampler(
                    analysis_interval_ms=1000.0 / cam_cfg.analysis_fps,
                    max_frame_age_seconds=cam_cfg.max_frame_age_seconds,
                    max_queue_size=cam_cfg.max_queue_size,
                ),
                change_detector=ChangeDetector(change_threshold=cam_cfg.change_threshold),
                analyzer=analyzer,
                deduplicator=EventDeduplicator(dedup_window_seconds=cam_cfg.dedup_window_seconds),
                bus=bus,
                session_id=session_id,
                capture_interval_ms=1000.0 / cam_cfg.capture_fps,
                analysis_interval_ms=1000.0 / cam_cfg.analysis_fps,
                idle=turns.idle if cfg.perception.pause_during_turn else None,
            )
            bus.subscribe(VisualEvent, _build_visual_attention_handler(
                attention, visual_context_ring, cfg.perception.max_visual_event_age_seconds, fusion,
            ))
            logger.info(
                "Visual perception configured — device=%d capture_fps=%.1f analysis_fps=%.1f",
                cam_cfg.device_index, cam_cfg.capture_fps, cam_cfg.analysis_fps,
            )

        # 6. Start the bus, then the mic: it must open before the MLX models
        # load to avoid a PortAudio device conflict on macOS Core Audio.
        await bus.start()
        cleanup.append(("bus", bus.shutdown))
        mic.start(loop)
        cleanup.append(("mic", mic.stop))

        def _open_tts() -> None:
            tts.load()
            player.start(tts.sample_rate)

        loads = [asyncio.to_thread(vad.load), asyncio.to_thread(asr.load)]
        if tts is not None:
            loads.append(asyncio.to_thread(_open_tts))
        if not await _until_shutdown(shutdown, asyncio.gather(*loads)):
            return

        if model is not None:
            # Daemon thread: warm-up must never delay startup or block exit.
            threading.Thread(target=_warm_up, args=(model, cfg), name="warm-up", daemon=True).start()

        if perception_pipeline is not None:
            await perception_pipeline.start()
            cleanup.append(("perception", perception_pipeline.stop))

        # Audio captured while the models loaded is stale.
        while mic.buffer.get_nowait() is not None:
            pass

        frontend_task = asyncio.create_task(frontend.run())

        async def _stop_frontend() -> None:
            frontend_task.cancel()
            for t in list(pending):
                t.cancel()
            await asyncio.gather(frontend_task, *pending, return_exceptions=True)
        cleanup.append(("frontend", _stop_frontend))

        logger.info("Pipeline running — speak into the microphone. Ctrl+C to stop.")
        await _until_shutdown(shutdown, frontend_task)
    finally:
        logger.info("Shutting down…")
        for name, fn in reversed(cleanup):
            try:
                result = fn()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Shutdown step %s failed", name)
        _log_session_summary(session_id, frontend, attention, recorder, player,
                             perception_pipeline, fusion, bus)


def _warm_up(model, cfg: JarvisConfig) -> None:
    # Only OllamaModel.warm_up reports reachability (False); there is no other host to try.
    if model.warm_up() is False:
        r = cfg.reasoning
        logger.error(
            "Reasoning host %s is unavailable; every reply will fail until it is up. "
            "Start it with `ollama serve` and pull the model with `ollama pull %s`.",
            r.ollama_url, r.model,
        )


def _log_session_summary(session_id, frontend, attention, recorder, player,
                         perception_pipeline, fusion, bus) -> None:
    logger.info("Session %s ended", session_id[:12])
    if frontend is not None:
        logger.info("Audio: %s", frontend.metrics)
    if attention is not None:
        logger.info("Attention counters: %s", attention.counters)
    outcomes = Counter(r["outcome"] for r in recorder.turns)
    logger.info("Turns: total=%d %s", len(recorder.turns), dict(outcomes))
    for stage, p in recorder.summary().items():
        if p.get("n"):
            logger.info("Latency %-16s n=%d p50=%.0fms p95=%.0fms max=%.0fms",
                        stage, p["n"], p["p50"], p["p95"], p["max"])
    if player is not None:
        logger.info("Playback: %s", player.stats)
    if bus is not None:
        logger.info("Events: published=%d dropped=%d", bus.events_published, bus.events_dropped)
    if perception_pipeline is not None:
        logger.info("Perception: %s", perception_pipeline.metrics)
    logger.info(
        "Visual attention: received=%d attention_worthy=%d",
        _metrics["visual_events_received"],
        _metrics["visual_events_attention_worthy"],
    )
    if fusion:
        logger.info("Fusion: %s", fusion.counters)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
