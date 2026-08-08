from app.events.types import Event, Modality, SpeechEvent


def test_event_has_unique_id():
    a, b = Event(), Event()
    assert a.id != b.id


def test_event_has_timestamp():
    e = Event()
    assert e.timestamp > 0


def test_event_session_and_correlation_defaults():
    e = Event()
    assert e.session_id == ""
    assert e.correlation_id is None


def test_event_source_default():
    e = Event()
    assert e.source == Modality.SYSTEM


def test_speech_event_fields():
    se = SpeechEvent(
        session_id="sess1",
        correlation_id="corr1",
        source=Modality.AUDIO,
        confidence=0.95,
        transcript="hello world",
        duration_seconds=1.5,
    )
    assert se.transcript == "hello world"
    assert se.duration_seconds == 1.5
    assert se.source == Modality.AUDIO
    assert se.session_id == "sess1"
    assert se.correlation_id == "corr1"
    assert isinstance(se, Event)


def test_speech_event_is_question_optional():
    se = SpeechEvent()
    assert se.is_question is False


def test_modality_values():
    assert Modality.AUDIO.name == "AUDIO"
    assert Modality.VISUAL.name == "VISUAL"
    assert Modality.SCREEN.name == "SCREEN"
    assert Modality.SYSTEM.name == "SYSTEM"
