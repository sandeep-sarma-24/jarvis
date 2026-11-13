# §T — Open Architectural Questions Requiring Decision

These are decisions that materially affect the architecture and cannot be resolved without your input or empirical testing.

## Classification

Questions are classified as:
- **BLOCKING** — must be resolved before Phase 1 begins
- **RESOLVE before Phase N** — must be resolved before that phase begins
- **DEFERRED** — explicitly deferred with a trigger phase for revisiting

No questions are BLOCKING for Phase 1. One question (T5) must be resolved before Phase 2.

## T1. TTS Voice Selection

**Question:** Kokoro-82M has 54 voices. The closest British options are `bf_emma` (British female) and potentially other `b*` prefixed voices. A male British voice (closer to MCU JARVIS / Paul Bettany's delivery) may not exist in the current Kokoro library.

**Options:**
- A: Use `bf_emma` (British female) — works immediately, high quality, but female voice diverges from MCU JARVIS.
- B: Use an American male voice from Kokoro (e.g., `am_adam`) and accept the accent difference.
- C: Upgrade to Qwen3-TTS/Holler for voice cloning — more work, but can synthesize any target voice. Delays Phase 4.
- D: Start with Kokoro (any suitable voice), plan Qwen3-TTS upgrade as Phase 4.5.

**Recommendation:** D — ship fast with the best available Kokoro voice, then invest in voice matching once the core system works.

**Resolved:** Kokoro was replaced by Phonon TTS (`hf://gradium/phonon-31m-en`) with the voice Easton (`tts.voice`); list voices with `.venv/bin/ptts --list-voices`.

**Classification: DEFERRED -> Phase 4**

## T2. Ollama vs LiteRT-LM for Gemma 4 12B Inference

**Question:** LiteRT-LM is Google's official runtime and shows better numbers on M4 Max. But benchmarks for 12B on base M4 (Air) are unconfirmed, and it's at v0.13. Ollama is battle-tested but slightly slower.

**Options:**
- A: Start with Ollama (safe, proven, ecosystem). Benchmark LiteRT-LM later.
- B: Start with LiteRT-LM (potentially faster). Accept newer, less-tested runtime.
- C: Abstract the interface and test both before Phase 3 begins.

**Recommendation:** A. Ollama gives us a working system faster. The `ReasoningModel` interface means swapping runtime is a config change.

**Resolved:** Ollama. The primary model is now `gemma4:e2b-it-qat` on a remote GPU host; `gemma4:12b` on local Ollama is the fallback. LiteRT-LM was not evaluated.

**Classification: DEFERRED -> benchmark during Phase 3**

## T3. Attention Classifier Training Strategy

**Question:** The Tier 2 classifier starts as a weighted feature combination (no ML). When should it become a trained model, and what's the training signal?

**Options:**
- A: Never — keep it as tunable weights. Simpler, no training data needed.
- B: After 100+ interactions with implicit feedback (user dismissed response = false positive). Train logistic regression.
- C: After 500+ interactions. Train a small neural net.

**Recommendation:** B. Logistic regression on implicit feedback is trivial to implement and provides real personalization. The interface doesn't change.

**Classification: DEFERRED -> revisit after sufficient interaction data (Phase 5+)**

## T4. Camera Capture on macOS Without Privacy Concerns

**Question:** macOS camera capture (via AVFoundation/OpenCV) shows a green indicator light. This is by design and non-negotiable. Users seeing the green light constantly may be uncomfortable.

**Impact:** Affects whether camera is always-on or triggered-on-demand.

**Options:**
- A: Camera is always-on when vision is enabled (green light always visible). Document that this is intentional.
- B: Camera activates only when triggered (e.g., user asks "what's on my desk?"). No persistent visual awareness.
- C: Periodic captures (every 5-10 seconds) — green light blinks briefly. Provides environmental awareness without persistent recording.

**Recommendation:** C for solo use. The periodic capture gives environmental awareness while being honest about when the camera is active. User should be able to set frequency or disable entirely.

**Classification: DEFERRED -> Phase 7**

## T5. How to Handle "Jarvis" in the Transcript

**Question:** The system has no wake word, but users will sometimes say "Jarvis" naturally. When the transcript contains "Jarvis," should it:
- Always respond (treating it as an explicit address)?
- Weigh it heavily but not absolutely (what if the user is talking *about* JARVIS to someone else)?
- Treat it like any other signal in the attention pipeline?

**Options:**
- A: "Jarvis" in transcript → always respond. Simple, reliable.
- B: "Jarvis" → high salience boost (e.g., +0.4) but still subject to full attention pipeline.
- C: "Jarvis" at start of utterance → always respond. "Jarvis" mid-sentence → boost only.

**Recommendation:** C. "Jarvis, what time is it?" is an explicit address. "I was telling Sarah about Jarvis earlier" is not. Position + context matters.

**Classification: RESOLVE before Phase 2** (affects attention rules implementation)

## T6. Screen Capture Permissions and Approach

**Question:** macOS screen capture requires explicit Screen Recording permission. The standard approach (CGWindowListCreateImage) captures the full screen. Alternative: use Accessibility APIs for just window titles + app names without the full screen capture permission.

**Options:**
- A: Full screen capture (requires Screen Recording permission). Full OCR capability.
- B: Accessibility API only (window titles, app names). No OCR, but no scary permission.
- C: Start with Accessibility API (Phase 9 MVP), add full capture as opt-in upgrade.

**Recommendation:** C. App name + window title gives 80% of the value for 0% of the privacy concern. Full OCR is an upgrade.

**Classification: DEFERRED -> Phase 9**

## T7. Response Length for Voice Output

**Question:** Voice responses should be concise (nobody wants a 200-word spoken essay). How short?

**Options:**
- A: Hard limit (max 2 sentences, ~30 words). Truncate longer responses.
- B: Soft guideline in system prompt ("respond in 1-2 sentences unless asked for detail").
- C: Adaptive — short for simple answers, longer for complex explanations, interrupt-able.

**Recommendation:** B for V1, C for V2. The system prompt can enforce brevity while allowing the model flexibility for genuinely complex answers.

**Classification: DEFERRED -> Phase 4** (TTS implementation)

---

# §U — Recommended MVP Definition

**MVP = Phase 0 through Phase 4 complete.**

A working MVP means: you speak to JARVIS, it decides whether to respond, and if so, it answers in voice. No memory persistence, no camera, no screen, no tools. Just the core loop:

```
You speak → VAD detects → ASR transcribes → Attention evaluates →
Context assembled → Gemma 4 reasons → Decision made → TTS speaks (or silence)
```

**What the MVP proves:**
1. The audio pipeline works in real-time on the M4 Air
2. The attention engine correctly filters most input to silence
3. The reasoning model produces useful, in-character responses
4. The TTS voice is acceptable
5. The total latency is <3.5 seconds (usable for conversation)
6. The memory usage fits in 16GB

**What the MVP doesn't have:**
- No persistence (restart = blank slate)
- No visual awareness
- No screen understanding
- No tools
- No proactive behavior
- Text-only fallback not yet implemented

**Estimated timeline:** 2-4 weeks of focused development (assuming models download quickly and macOS permissions cooperate).

---

# §V — Definition of "V1 JARVIS"

**V1 = Phases 0 through 6 + Phase 12 (privacy).**

V1 is a voice assistant that listens, answers, and remembers — locally, privately, in character.

**V1 capabilities:**
- Continuous audio perception with intelligent attention filtering
- Spoken responses in JARVIS personality
- Working memory (conversation context within a session)
- Persistent episodic and semantic memory (remembers across restarts)
- User preferences learned and recalled
- Privacy controls (pause, privacy mode, memory viewing/deletion)
- Structured logging and basic observability
- Stable enough for daily use (hours-long sessions without crashes)

**V1 experience:** You work at your desk. JARVIS listens. Most of the time it's silent. When you ask a question — directly or in a way that's clearly meant for it — it answers in a composed British voice. It remembers what you told it yesterday. It knows your preferences. It never interrupts when you're talking to someone else.

**V1 does NOT:**
- See (no camera, no screen awareness)
- Execute tools or take actions
- Proactively offer help
- Understand what's on your screen

---

# §W — Definition of "V2 JARVIS"

**V2 = All phases through Phase 11 (proactive intelligence).**

V2 is a multimodal assistant that perceives, remembers, and occasionally acts — with awareness of your visual environment and workflow.

**V2 adds over V1:**
- Camera perception (knows what's on your desk, whether you're present)
- Screen understanding (knows what app you're using, what you're working on)
- Multimodal context (combines audio + visual + screen for richer understanding)
- Tool execution (can search the web, check files, set reminders, run commands)
- Proactive intelligence (occasionally offers help unprompted, when confident it's useful)
- Voice matching (JARVIS voice closer to target via voice cloning)

**V2 experience:** You're debugging an API at 11pm. JARVIS sees VS Code on screen, hears you mutter "why is this returning 500," recalls that you were working on the same API yesterday, and says: "Sir, the auth endpoint was returning 500 yesterday due to an expired token. Shall I check the token configuration?" You didn't ask. But it was right, and it was helpful.

---

# §X — Definition of "Fully Realized System"

The fully realized JARVIS is an always-present personal intelligence layer that perceives all modalities, maintains rich persistent memory, executes complex tool chains, learns from every interaction, and integrates seamlessly into the user's life.

**Beyond V2:**
- Multi-room / multi-device awareness
- Calendar and schedule integration
- Email and communication management
- Proactive task management (notices patterns, suggests automation)
- Long-term goal tracking and progress awareness
- Social context (knows who's in the room, adjusts behavior)
- Deep workflow integration (IDE plugins, browser extensions)
- Self-improvement (evaluates its own performance, tunes thresholds)
- Graceful cloud fallback for tasks that exceed local capability
- Mobile presence (phone notifications, remote queries)

**Fully realized experience:** JARVIS knows your schedule, your projects, your preferences, your habits. It manages your home, reminds you of things you'd forget, handles routine communications, and is genuinely useful dozens of times per day — while never being annoying, never overstepping, and never compromising your privacy without explicit consent.

This is years of development. The architecture supports it. The phased approach gets there incrementally, with each phase delivering real value.

---

## Principles Self-Check (Architectural Principle 16)

All 15 principles reviewed against the full document:

| # | Principle | Addressed | Notes |
|---|-----------|-----------|-------|
| 1 | Perception is continuous | ✓ | VAD always running, adaptive frame sampling |
| 2 | Expensive reasoning is selective | ✓ | Tiered attention gates LLM access |
| 3 | Memory is external to LLM context | ✓ | SQLite store, retrieval populates context |
| 4 | Context is dynamically constructed | ✓ | Context builder assembles per-event |
| 5 | Models must be replaceable | ✓ | Abstract interfaces for all model tiers |
| 6 | Components independently testable | ✓ | Interface-driven design, mock-friendly |
| 7 | Privacy designed from beginning | ✓ | Permission model, privacy mode, no raw storage |
| 8 | Works locally | ✓ | All inference on-device, SQLite storage |
| 9 | Avoid unnecessary infrastructure | ✓ | No Redis, no NATS, no Docker, no external DB |
| 10 | Optimize only after measuring | ✓ | Phase 14 follows Phase 13 (evaluation) |
| 11 | Every subsystem needs an interface | ✓ | §L defines all interfaces |
| 12 | Measurable acceptance criteria | ✓ | Phases 1-3 have full criteria; others get them JIT |
| 13 | Don't over-engineer V1 | ✓ | Weighted scoring over ML classifier; SQLite over Postgres; asyncio over message queue |
| 14 | System can remain silent | ✓ | Silence is default; attention is conservative |
| 15 | Useful through memory, not context size | ✓ | 3K token target context; memory retrieval fills it |
| 16 | World State is distinct from persistent memory | ✓ | World State = now; Memory = before; separate subsystems |
| 17 | Attention determines whether reasoning resources are spent | ✓ | Attention answers "should reason?" only |
| 18 | Response decision is separate from attention | ✓ | Reasoning may fire without responding |
| 19 | Perception events are correlatable | ✓ | Events carry session_id and correlation_id |
| 20 | Models are replaceable behind interfaces | ✓ | All models sit behind abstract interfaces |
| 21 | Performance budgets are measured targets, not guarantees | ✓ | Validated empirically, not architectural blockers |

**Tradeoffs explicitly made:**
- **Principle 8 (local) vs. reasoning quality:** Accepted. Gemma 4 E2B (and the 12B fallback) are not GPT-4-class. Reasoning now runs on a second machine on the local network, still not a cloud service. Cloud fallback is the documented upgrade path.
- **Principle 9 (no infra) vs. author's NATS/Redis familiarity:** In-process solutions win for single-user/single-machine. The familiarity argument doesn't outweigh the "no failure modes" argument for a personal tool. Redis/NATS are the documented upgrade path if multi-process becomes necessary.
- **Principle 13 (don't over-engineer) vs. Principle 5 (models replaceable):** Both honored. Interfaces add no runtime cost and minimal code — they're just abstract base classes. The implementations behind them are as simple as possible.
