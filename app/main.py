"""JARVIS — Multimodal perception → Fusion → Attention → Context → LLM → Response Decision → TTS.

Entry point and lifecycle orchestrator.
Phase 1 audio + Phase 2 attention + Phase 3 reasoning + Phase 4 TTS + Phase 5 memory
+ Phase 6 visual perception + Phase 8 multimodal fusion.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from collections import deque
from pathlib import Path

from app.config import JarvisConfig, load_config
from app.events.bus import EventBus
from app.events.types import Event, Modality, ResponseEvent, SpeechEvent, VisualEvent
from app.audio.capture import MicrophoneCapture
from app.audio.vad import VoiceActivityDetector
from app.audio.segmentation import SpeechSegmenter
from app.audio.asr import SpeechRecognizer
from app.attention.engine import AttentionEngine
from app.fusion.engine import MultimodalFusion
from app.memory.working import WorkingMemory

logger = logging.getLogger("jarvis")

# ── Metrics ──────────────────────────────────────────────────────────────

_metrics = {
    "audio_chunks_received": 0,
    "audio_chunks_dropped": 0,
    "speech_segments_detected": 0,
    "speech_segments_transcribed": 0,
    "speech_segments_failed": 0,
    "events_published": 0,
    "events_dropped": 0,
    "buffer_max_depth": 0,
    "reasoning_requests": 0,
    "reasoning_successes": 0,
    "reasoning_failures": 0,
    "reasoning_dropped": 0,
    "responses_generated": 0,
    "tts_requests": 0,
    "tts_successes": 0,
    "tts_failures": 0,
    "tts_latency_ms": 0.0,
    "playback_started": 0,
    "playback_completed": 0,
    "playback_interrupted": 0,
    "visual_events_received": 0,
    "visual_events_attention_worthy": 0,
    "fusion_events_received": 0,
    "fusion_events_correlated": 0,
    "fusion_events_expired": 0,
    "fusion_events_dropped": 0,
    "fusion_snapshots_created": 0,
    "fusion_out_of_order_events": 0,
    "fusion_context_characters": 0,
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


# ── Pipeline ─────────────────────────────────────────────────────────────

async def _audio_pipeline(
    mic: MicrophoneCapture,
    vad: VoiceActivityDetector,
    segmenter: SpeechSegmenter,
    asr: SpeechRecognizer,
    bus: EventBus,
    session_id: str,
) -> None:
    """Async loop: read chunks → VAD → segment → ASR → publish SpeechEvent."""
    while True:
        chunk = await mic.buffer.get()
        _metrics["audio_chunks_received"] = mic.buffer.chunks_received
        _metrics["audio_chunks_dropped"] = mic.buffer.chunks_dropped
        _metrics["buffer_max_depth"] = mic.buffer.max_depth

        vad_result = vad.process_chunk(chunk)

        segment = segmenter.process(chunk, vad_result)
        if segment is None:
            continue

        _metrics["speech_segments_detected"] = segmenter.segments_detected

        depth_before = mic.buffer.depth
        drops_before = mic.buffer.chunks_dropped

        try:
            asr_result = await asr.transcribe_async(segment.audio, segment.sample_rate)
        except Exception:
            _metrics["speech_segments_failed"] += 1
            logger.exception("ASR failed on segment (%.2fs)", segment.duration_seconds)
            continue

        depth_after = mic.buffer.depth
        drops_after = mic.buffer.chunks_dropped

        if not asr_result.text:
            continue

        _metrics["speech_segments_transcribed"] += 1

        total_ms = (time.time() - segment.end_time) * 1000

        event = SpeechEvent(
            session_id=session_id,
            correlation_id=uuid.uuid4().hex,
            source=Modality.AUDIO,
            confidence=0.0,
            transcript=asr_result.text,
            duration_seconds=segment.duration_seconds,
            metadata={
                "asr_latency_ms": round(asr_result.latency_ms, 1),
                "total_speech_to_event_ms": round(total_ms, 1),
                "buffer_depth_before_asr": depth_before,
                "buffer_depth_after_asr": depth_after,
                "chunks_dropped_during_asr": drops_after - drops_before,
            },
        )

        await bus.publish(event)
        _metrics["events_published"] = bus.events_published
        _metrics["events_dropped"] = bus.events_dropped


# ── Multimodal context formatting ─────────────────────────────────────────

def _format_multimodal_context(snap) -> list[str] | None:
    """Format a fusion snapshot into context strings for ContextBuilder.

    Returns structured observations tagged by modality, most recent last.
    The final visual observation is marked as current to help the reasoning
    model prioritize it over older (potentially stale) descriptions.
    """
    parts: list[str] = []

    for obs in snap.audio_observations:
        parts.append(f"[audio] {obs}")

    visuals = snap.visual_observations
    for obs in visuals[:-1]:
        parts.append(f"[visual] {obs}")
    if visuals:
        parts.append(f"[visual — current] {visuals[-1]}")

    return parts if parts else None


# ── Reasoning pipeline ──────────────────────────────────────────────────

def _build_reasoning_handler(
    cfg: JarvisConfig,
    attention: AttentionEngine,
    bus: EventBus,
    memory_manager=None,
    visual_context_ring: deque | None = None,
    fusion: MultimodalFusion | None = None,
    on_demand_vision=None,
):
    """Build the attention→context→reasoning→response handler.

    Returns the async handler and the model instance (for cleanup).
    Uses a bounded semaphore to limit concurrent reasoning.
    Phase 5: integrates conversation state and memory retrieval.
    Phase 6: includes recent visual context.
    Phase 8: uses fused multimodal context when fusion is enabled.

    on_demand_vision: optional (camera, analyzer) tuple for live frame capture
    when the user asks a vision question.
    """
    from app.context.builder import ContextBuilder
    from app.reasoning.local_model import OllamaModel
    from app.reasoning.model import ReasoningModel
    from app.response.decision import decide

    model: ReasoningModel = OllamaModel(
        model=cfg.reasoning.model,
        base_url=cfg.reasoning.ollama_url,
        timeout_seconds=cfg.reasoning.timeout_seconds,
    )

    context_builder = ContextBuilder(
        max_messages=cfg.reasoning.max_context_messages,
        max_characters=cfg.reasoning.max_context_characters,
        max_memory_items=cfg.memory.max_retrieved_memories,
        max_visual_events=cfg.perception.max_visual_events,
        max_visual_context_characters=cfg.perception.max_visual_context_characters,
    )

    reasoning_sem = asyncio.Semaphore(cfg.reasoning.max_concurrent_requests)
    reasoning_logger = logging.getLogger("jarvis.reasoning")

    async def _handler(event: Event) -> None:
        if not isinstance(event, SpeechEvent):
            return

        decision = attention.evaluate(event)

        if decision.should_reason:
            print(
                f"\n[ATTENTION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  should_reason=true\n"
                f"  salience={decision.salience:.2f}\n"
            )
        else:
            print(
                f"\n[ATTENTION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  should_reason=false\n"
                f"\n[NO RESPONSE]\n"
            )
            return

        if not cfg.reasoning.enabled or not cfg.response.enabled:
            return

        if reasoning_sem.locked():
            _metrics["reasoning_dropped"] += 1
            reasoning_logger.warning(
                "Reasoning slot full, dropping event %s", event.id[:12]
            )
            return

        _metrics["reasoning_requests"] += 1

        # Phase 5: persist user message
        conversation = memory_manager.conversation if memory_manager else None
        if conversation and not conversation.has_message_for_event(event.id):
            await asyncio.to_thread(
                conversation.add_user_message, event.transcript, event.id
            )

        async with reasoning_sem:
            # Phase 5: gather conversation history and relevant memories
            conv_messages: list[dict[str, str]] | None = None
            mem_context: list[str] | None = None

            if conversation and cfg.memory.enabled:
                recent = await asyncio.to_thread(
                    conversation.recent_messages, cfg.memory.max_recent_messages
                )
                # Exclude the current message from history
                conv_messages = [
                    {"role": m.role, "content": m.content}
                    for m in recent
                    if m.source_event_id != event.id
                ]

                # Retrieve relevant memories
                if memory_manager:
                    memories = await asyncio.to_thread(
                        memory_manager.search, event.transcript, cfg.memory.max_retrieved_memories
                    )
                    if memories:
                        mem_context = [m.content for m in memories]

            # Phase 8: use fused multimodal context; ring as visual fallback
            multimodal_ctx: list[str] | None = None
            vis_ctx: list[str] | None = None

            # On-demand vision: if user asks about what they see, grab a fresh frame
            _VISION_KEYWORDS = ("see", "look", "watch", "visual", "camera", "showing")
            is_vision_request = any(kw in event.transcript.lower() for kw in _VISION_KEYWORDS)

            if is_vision_request and on_demand_vision is not None and on_demand_vision[0] is not None:
                camera, analyzer = on_demand_vision[0]
                frame = await camera.capture()
                if frame is not None:
                    analysis = await analyzer.analyze(frame)
                    if analysis is not None:
                        multimodal_ctx = [f"[visual — live] {analysis.description}"]
                        reasoning_logger.info(
                            "ON_DEMAND_VISION desc=%r", analysis.description
                        )

            if fusion and fusion.enabled:
                snap = fusion.snapshot(
                    exclude_event_ids={event.id},
                )
                fusion_ctx = _format_multimodal_context(snap)
                if fusion_ctx:
                    multimodal_ctx = (multimodal_ctx or []) + fusion_ctx
                # Fusion snapshot may have visual=0 if vision analysis is still
                # in-flight. Fall back to the ring which captures attention-worthy
                # visual events as they complete (independent of snapshot timing).
                if not snap.visual_observations and visual_context_ring:
                    for desc in visual_context_ring:
                        multimodal_ctx = multimodal_ctx or []
                        multimodal_ctx.append(f"[visual] {desc[1]}")
                _metrics["fusion_context_characters"] = snap.total_characters
                reasoning_logger.info(
                    "FUSION_SNAPSHOT audio=%d visual=%d chars=%d",
                    len(snap.audio_observations),
                    len(snap.visual_observations),
                    snap.total_characters,
                )
            elif visual_context_ring:
                vis_ctx = [desc for _, desc in visual_context_ring]

            ctx = context_builder.build(
                event, attention.working_memory,
                conversation_messages=conv_messages,
                memory_context=mem_context,
                visual_context=vis_ctx,
                multimodal_context=multimodal_ctx,
            )
            messages = context_builder.to_messages(ctx)

            reasoning_logger.debug(
                "Context: %d messages, %d chars",
                len(messages), ctx.total_characters,
            )

            result = await asyncio.to_thread(
                model.generate,
                messages,
                max_tokens=cfg.reasoning.max_output_tokens,
                temperature=cfg.reasoning.temperature,
            )

        if result.success:
            _metrics["reasoning_successes"] += 1
        else:
            _metrics["reasoning_failures"] += 1

        reasoning_logger.info(
            "REASONING event_id=%s model=%s latency_ms=%.0f success=%s "
            "context_messages=%d context_chars=%d",
            event.id[:12], result.model, result.latency_ms, result.success,
            len(messages), ctx.total_characters,
        )

        resp = decide(event.id, result)

        if resp.respond and resp.text:
            _metrics["responses_generated"] += 1

            # Phase 5: persist assistant response before publishing
            if conversation:
                await asyncio.to_thread(
                    conversation.add_assistant_message, resp.text, event.id + "_response"
                )

            thinking_block = ""
            if resp.thinking:
                thinking_block = f"\n[THINKING]\n  {resp.thinking}\n"

            print(
                f"\n[REASONING]\n"
                f"  model={result.model}\n"
                f"  latency={result.latency_ms:.0f}ms\n"
                f"{thinking_block}"
                f"\n[RESPONSE]\n"
                f"  {resp.text}\n"
            )

            response_event = ResponseEvent(
                session_id=event.session_id,
                correlation_id=event.correlation_id,
                source=Modality.SYSTEM,
                event_id=event.id,
                text=resp.text,
                reasoning_latency_ms=result.latency_ms,
                speech_end_timestamp=event.timestamp,
            )
            await bus.publish(response_event)
        else:
            print(
                f"\n[RESPONSE_DECISION]\n"
                f"  event_id={event.id[:12]}\n"
                f"  respond=false\n"
                f"  reason=\"{resp.reason}\"\n"
            )

        reasoning_logger.info(
            "RESPONSE_DECISION event_id=%s respond=%s reason=%r",
            event.id[:12], resp.respond, resp.reason,
        )

    return _handler, model


# ── TTS pipeline ──────────────────────────────────────────────────────

def _build_tts_handler(cfg: JarvisConfig, mic: MicrophoneCapture | None = None):
    """Build the TTS subscriber: ResponseEvent → synthesize → play.

    Returns the async handler, TTS engine, and player (for cleanup).
    Text exceeding max_text_length is split at sentence boundaries.
    When mic is provided, suppresses capture during playback to prevent
    the ASR from transcribing TTS output (feedback loop).
    """
    from app.tts.local_tts import PhononTTS
    from app.tts.player import LocalAudioPlayer, PlaybackOutcome

    mic_ref: list[MicrophoneCapture | None] = [mic]

    def _on_playback_done(outcome: PlaybackOutcome) -> None:
        if mic_ref[0] is not None:
            mic_ref[0].suppressed = False
        if outcome == PlaybackOutcome.COMPLETED:
            _metrics["playback_completed"] += 1
        elif outcome == PlaybackOutcome.INTERRUPTED:
            _metrics["playback_interrupted"] += 1

    tts_engine = PhononTTS(
        model_id=cfg.tts.model,
        voice=cfg.tts.voice,
        quant=cfg.tts.quant,
    )

    player = LocalAudioPlayer(volume=cfg.playback.volume, on_done=_on_playback_done)
    tts_logger = logging.getLogger("jarvis.tts")

    def _truncate_text(text: str) -> str:
        """Enforce max_text_length. Split at sentence boundary if possible."""
        limit = cfg.tts.max_text_length
        if len(text) <= limit:
            return text
        for i in range(limit - 1, max(limit // 2, 0), -1):
            if text[i] in ".!?\n":
                return text[: i + 1]
        return text[:limit]

    async def _handler(event: Event) -> None:
        if not isinstance(event, ResponseEvent):
            return

        if not cfg.tts.enabled or not cfg.playback.enabled:
            return

        text = _truncate_text(event.text)
        if not text.strip():
            return

        _metrics["tts_requests"] += 1
        tts_logger.info("TTS event_id=%s text_len=%d", event.event_id[:12], len(text))

        t0 = time.time()
        result = await tts_engine.synthesize_async(text)
        tts_latency = (time.time() - t0) * 1000
        _metrics["tts_latency_ms"] = round(tts_latency, 1)

        if not result.success or result.audio is None:
            _metrics["tts_failures"] += 1
            tts_logger.error(
                "TTS failed event_id=%s error=%s", event.event_id[:12], result.error
            )
            return

        _metrics["tts_successes"] += 1

        # speech_end_to_audio_start: time from original speech end to playback start
        # This includes: attention + reasoning + TTS synthesis
        speech_end_to_audio_ms = (
            (time.time() - event.speech_end_timestamp) * 1000
            if event.speech_end_timestamp > 0
            else 0.0
        )

        # response_event_to_audio: time from ResponseEvent creation to playback
        response_to_audio_ms = (time.time() - event.timestamp) * 1000

        tts_logger.info(
            "TTS event_id=%s model=%s tts_latency_ms=%.0f duration=%.2fs "
            "speech_end_to_audio_ms=%.0f response_to_audio_ms=%.0f",
            event.event_id[:12], result.model, result.latency_ms,
            result.duration_seconds, speech_end_to_audio_ms, response_to_audio_ms,
        )

        _metrics["playback_started"] += 1
        if mic_ref[0] is not None:
            mic_ref[0].suppressed = True
        player.play(result.audio, result.sample_rate)

        print(
            f"\n[TTS]\n"
            f"  event_id={event.event_id[:12]}\n"
            f"  tts_latency={result.latency_ms:.0f}ms\n"
            f"  duration={result.duration_seconds:.2f}s\n"
            f"  speech_end_to_audio={speech_end_to_audio_ms:.0f}ms\n"
            f"  model={result.model}\n"
        )

    return _handler, tts_engine, player, mic_ref


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

async def run(cfg: JarvisConfig | None = None) -> None:
    cfg = cfg or load_config()

    logging.basicConfig(
        level=getattr(logging, cfg.logging.level, logging.INFO),
        format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    session_id = uuid.uuid4().hex
    logger.info("JARVIS session %s starting", session_id[:12])
    logger.info("Microphone capture will be active — audio is processed locally, never stored.")

    # 0. Memory (Phase 5)
    memory_manager = None
    if cfg.memory.enabled:
        from app.memory.manager import MemoryManager
        from app.memory.sqlite_store import SQLiteStore

        db_path = Path(cfg.memory.database_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = SQLiteStore(db_path)
        memory_manager = MemoryManager(store)
        memory_manager.start_session()
        logger.info(
            "Memory active — db=%s session=%s",
            db_path, memory_manager.conversation.session_id[:12],
        )

    # 1. EventBus
    bus = EventBus(queue_size=cfg.event_bus.subscriber_queue_size)
    bus.subscribe(SpeechEvent, _console_handler)

    # 2. Visual context ring (must exist before reasoning handler references it)
    visual_context_ring: deque[tuple[float, str]] = deque(maxlen=cfg.perception.max_visual_events)

    # 2a. Multimodal fusion (Phase 8)
    fusion: MultimodalFusion | None = None
    if cfg.fusion.enabled:
        fusion = MultimodalFusion(cfg.fusion)
        logger.info(
            "Fusion active — window=%.1fs max_events=%d max_chars=%d",
            cfg.fusion.window_seconds, cfg.fusion.max_events, cfg.fusion.max_characters,
        )

    # 3. Attention + Reasoning pipeline (Phase 2 + 3)
    attention: AttentionEngine | None = None
    model_cleanup = None
    tts_cleanup = None
    player_cleanup = None
    vision_analyzer_cleanup = None

    # Mutable ref for on-demand vision (populated after camera/analyzer are created)
    on_demand_vision_ref: list = [None]

    if cfg.attention.enabled:
        working_mem = WorkingMemory(max_size=cfg.attention.working_memory_size)
        attention = AttentionEngine(
            assistant_names=cfg.attention.assistant_names,
            reason_threshold=cfg.attention.reason_threshold,
            working_memory=working_mem,
        )

        if cfg.reasoning.enabled:
            handler, model_cleanup = _build_reasoning_handler(
                cfg, attention, bus, memory_manager, visual_context_ring, fusion,
                on_demand_vision=on_demand_vision_ref,
            )
            bus.subscribe(SpeechEvent, handler)
            logger.info(
                "Reasoning pipeline active — model=%s max_concurrent=%d",
                cfg.reasoning.model, cfg.reasoning.max_concurrent_requests,
            )
        else:
            async def _attention_only_handler(event: Event) -> None:
                if isinstance(event, SpeechEvent) and attention is not None:
                    attention.evaluate(event)
            bus.subscribe(SpeechEvent, _attention_only_handler)

        logger.info("Attention engine active — names=%s threshold=%.2f",
                     cfg.attention.assistant_names, cfg.attention.reason_threshold)

    # 3a. Fusion speech ingestion (Phase 8) — all speech enters fusion
    if fusion:
        async def _fusion_speech_handler(event: Event) -> None:
            if isinstance(event, SpeechEvent):
                fusion.ingest(event)
        bus.subscribe(SpeechEvent, _fusion_speech_handler)

    # 4. TTS pipeline (Phase 4)
    tts_mic_ref = None
    if cfg.tts.enabled and cfg.playback.enabled:
        tts_handler, tts_cleanup, player_cleanup, tts_mic_ref = _build_tts_handler(cfg)
        bus.subscribe(ResponseEvent, tts_handler)
        logger.info(
            "TTS pipeline active — model=%s voice=%s",
            cfg.tts.model, cfg.tts.voice,
        )

    # 5. Visual perception pipeline (Phase 6)
    perception_pipeline = None

    if cfg.perception.enabled and cfg.perception.camera.enabled and attention is not None:
        from app.perception.camera import CameraError, OpenCVCamera
        from app.perception.pipeline import PerceptionPipeline
        from app.perception.sampler import FrameSampler
        from app.perception.visual import ChangeDetector, EventDeduplicator, MockVisualAnalyzer

        cam_cfg = cfg.perception.camera
        camera = OpenCVCamera(
            device_index=cam_cfg.device_index,
            width=cam_cfg.width,
            height=cam_cfg.height,
        )
        sampler = FrameSampler(
            analysis_interval_ms=1000.0 / cam_cfg.analysis_fps,
            max_frame_age_seconds=cam_cfg.max_frame_age_seconds,
            max_queue_size=cam_cfg.max_queue_size,
        )
        change_detector = ChangeDetector(change_threshold=cam_cfg.change_threshold)

        vis_cfg = cfg.perception.vision
        if vis_cfg.enabled:
            if vis_cfg.provider != "ollama":
                raise ValueError(
                    f"Unsupported vision provider: {vis_cfg.provider!r} (only 'ollama' is supported)"
                )

            from app.perception.local_vision import LocalVisionAnalyzer, OllamaVisionBackend

            vision_backend = OllamaVisionBackend(
                model=vis_cfg.model,
                base_url=vis_cfg.ollama_url,
                timeout_seconds=vis_cfg.timeout_seconds,
                temperature=vis_cfg.temperature,
            )
            analyzer = LocalVisionAnalyzer(
                backend=vision_backend,
                prompt=vis_cfg.prompt,
                timeout_seconds=vis_cfg.timeout_seconds,
                max_image_size=vis_cfg.max_image_size,
            )
            vision_analyzer_cleanup = analyzer
            on_demand_vision_ref[0] = (camera, analyzer)
            logger.info("Vision analyzer: LocalVisionAnalyzer (model=%s)", vis_cfg.model)
        else:
            analyzer = MockVisualAnalyzer()
            logger.info("Vision analyzer: MockVisualAnalyzer")

        deduplicator = EventDeduplicator(dedup_window_seconds=cam_cfg.dedup_window_seconds)

        analysis_interval_ms = 1000.0 / cam_cfg.analysis_fps
        perception_pipeline = PerceptionPipeline(
            camera=camera,
            sampler=sampler,
            change_detector=change_detector,
            analyzer=analyzer,
            deduplicator=deduplicator,
            bus=bus,
            session_id=session_id,
            capture_interval_ms=1000.0 / cam_cfg.capture_fps,
            analysis_interval_ms=analysis_interval_ms,
        )

        vis_handler = _build_visual_attention_handler(
            attention, visual_context_ring, cfg.perception.max_visual_event_age_seconds, fusion,
        )
        bus.subscribe(VisualEvent, vis_handler)
        logger.info(
            "Visual perception configured — device=%d capture_fps=%.1f analysis_fps=%.1f",
            cam_cfg.device_index, cam_cfg.capture_fps, cam_cfg.analysis_fps,
        )

    await bus.start()

    # 5a. Start perception pipeline if configured
    if perception_pipeline is not None:
        await perception_pipeline.start()

    # 6. Microphone — must open before MLX model loads (ASR/TTS) to avoid
    # PortAudio device conflict on macOS Core Audio.
    mic = MicrophoneCapture(cfg.audio)
    if tts_mic_ref is not None:
        tts_mic_ref[0] = mic
    loop = asyncio.get_running_loop()

    shutdown = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    mic.start(loop)

    # 7. VAD
    vad = VoiceActivityDetector(threshold=cfg.vad.threshold)
    vad.load()

    # 8. ASR (MLX — loads on dedicated thread)
    asr = SpeechRecognizer(cfg.asr)
    asr.load()

    # 8a. Pre-load TTS model on its dedicated thread
    if tts_cleanup is not None:
        tts_cleanup.load()

    # 9. Segmenter
    segmenter = SpeechSegmenter(
        config=cfg.vad,
        sample_rate=cfg.audio.sample_rate,
        chunk_samples=cfg.audio.chunk_samples,
    )

    # 10. Run audio pipeline
    pipeline_task = asyncio.create_task(
        _audio_pipeline(mic, vad, segmenter, asr, bus, session_id)
    )

    logger.info("Pipeline running — speak into the microphone. Ctrl+C to stop.")

    await shutdown.wait()

    # 11. Graceful shutdown
    logger.info("Shutting down…")

    # Stop perception before audio (reverse startup order)
    if perception_pipeline is not None:
        await perception_pipeline.stop()

    mic.stop()
    pipeline_task.cancel()
    try:
        await pipeline_task
    except asyncio.CancelledError:
        pass
    await bus.shutdown()
    asr.close()

    if vision_analyzer_cleanup is not None:
        vision_analyzer_cleanup.close()
    if player_cleanup is not None:
        player_cleanup.close()
    if tts_cleanup is not None:
        tts_cleanup.close()
    if model_cleanup is not None:
        model_cleanup.close()
    if memory_manager is not None:
        memory_manager.close()

    logger.info(
        "Session %s ended — chunks=%d dropped=%d segments=%d transcribed=%d failed=%d max_depth=%d",
        session_id[:12],
        _metrics["audio_chunks_received"],
        _metrics["audio_chunks_dropped"],
        _metrics["speech_segments_detected"],
        _metrics["speech_segments_transcribed"],
        _metrics["speech_segments_failed"],
        _metrics["buffer_max_depth"],
    )
    if attention is not None:
        logger.info("Attention counters: %s", attention.counters)
    logger.info(
        "Reasoning: requests=%d success=%d fail=%d dropped=%d responses=%d",
        _metrics["reasoning_requests"],
        _metrics["reasoning_successes"],
        _metrics["reasoning_failures"],
        _metrics["reasoning_dropped"],
        _metrics["responses_generated"],
    )
    logger.info(
        "TTS: requests=%d success=%d fail=%d latency=%.0fms "
        "playback_started=%d completed=%d interrupted=%d",
        _metrics["tts_requests"],
        _metrics["tts_successes"],
        _metrics["tts_failures"],
        _metrics["tts_latency_ms"],
        _metrics["playback_started"],
        _metrics["playback_completed"],
        _metrics["playback_interrupted"],
    )
    if perception_pipeline is not None:
        logger.info("Perception: %s", perception_pipeline.metrics)
    logger.info(
        "Visual attention: received=%d attention_worthy=%d",
        _metrics["visual_events_received"],
        _metrics["visual_events_attention_worthy"],
    )
    if fusion:
        fc = fusion.counters
        _metrics.update({f"fusion_{k}": v for k, v in fc.items()})
        logger.info("Fusion: %s", fc)


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
