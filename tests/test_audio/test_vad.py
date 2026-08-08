import numpy as np
import pytest

from app.audio.vad import VoiceActivityDetector, VADResult


@pytest.mark.integration
def test_vad_loads_and_processes():
    """Requires Silero VAD model files. Run with: pytest -m integration"""
    vad = VoiceActivityDetector(threshold=0.5)
    vad.load()

    silence = np.zeros(512, dtype=np.float32)
    result = vad.process_chunk(silence)
    assert isinstance(result, VADResult)
    assert result.latency_ms >= 0
    assert not result.is_speech

    vad.reset()


def test_vad_not_loaded_raises():
    vad = VoiceActivityDetector()
    with pytest.raises(RuntimeError, match="not loaded"):
        vad.process_chunk(np.zeros(512, dtype=np.float32))
