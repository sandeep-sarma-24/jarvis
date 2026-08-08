import numpy as np
import pytest

from app.audio.asr import SpeechRecognizer, ASRResult
from app.config import ASRConfig


@pytest.mark.integration
def test_asr_loads_and_transcribes():
    """Requires Qwen3-ASR model. Run with: pytest -m integration"""
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
