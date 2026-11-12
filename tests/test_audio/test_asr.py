import numpy as np
import pytest

from app.audio.asr import SpeechRecognizer, ASRResult
from app.config import ASRConfig


@pytest.mark.integration
def test_asr_loads_and_transcribes():
    """Requires Granite Speech model. Run with: pytest -m integration"""
    asr = SpeechRecognizer(ASRConfig())
    asr.load()

    # 2 seconds of silence — should produce empty or minimal transcript
    audio = np.zeros(32000, dtype=np.float32)
    result = asr.transcribe(audio, sample_rate=16000)
    assert isinstance(result, ASRResult)
    assert result.latency_ms >= 0


def test_asr_not_loaded_raises():
    asr = SpeechRecognizer(ASRConfig())
    with pytest.raises(RuntimeError, match="not loaded"):
        asr.transcribe(np.zeros(16000, dtype=np.float32))


@pytest.mark.integration
def test_tts_to_asr_round_trip():
    """Phonon speaks a sentence, Granite transcribes it back."""
    from app.tts.local_tts import PhononTTS

    tts = PhononTTS()
    spoken = tts.synthesize("Good evening. All systems are online.")
    tts.close()
    assert spoken.success, spoken.error
    # 24 kHz → 16 kHz linear resample is fine for intelligibility
    audio = np.interp(
        np.arange(0, len(spoken.audio), spoken.sample_rate / 16000),
        np.arange(len(spoken.audio)), spoken.audio,
    ).astype(np.float32)

    asr = SpeechRecognizer(ASRConfig())
    asr.load()
    text = asr._executor.submit(asr.transcribe, audio).result().text.lower()
    asr.close()
    assert "all systems are online" in text
