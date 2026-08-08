"""Tests for Tier 2 attention scorer."""
import pytest

from app.attention.classifier import AttentionScorer, ScorerWeights


@pytest.fixture
def scorer():
    return AttentionScorer()


class TestScorerBasics:

    def test_empty_transcript(self, scorer):
        salience, features = scorer.score("")
        assert salience == 0.0
        assert features.question == 0.0

    def test_salience_bounded_0_1(self, scorer):
        salience, _ = scorer.score("What is the meaning of life?")
        assert 0.0 <= salience <= 1.0

    def test_background_lowers_salience(self, scorer):
        bg_salience, _ = scorer.score("I'll call you tomorrow.")
        q_salience, _ = scorer.score("Why is the API failing?")
        assert bg_salience < q_salience


class TestFeatureExtraction:

    def test_question_feature(self, scorer):
        _, features = scorer.score("What time is it?")
        assert features.question > 0.3

    def test_request_feature(self, scorer):
        _, features = scorer.score("Can you explain this error?")
        assert features.request_language > 0.0

    def test_technical_terms(self, scorer):
        _, features = scorer.score("The API endpoint is returning a 500 error from the database query.")
        assert features.technical_terms > 0.0

    def test_background_signal(self, scorer):
        _, features = scorer.score("I'll call you tomorrow.")
        assert features.background_signal > 0.0

    def test_explicit_address_flag(self, scorer):
        _, features = scorer.score("What time?", is_addressed=True)
        assert features.explicit_address == 1.0

    def test_transcript_length(self, scorer):
        _, short_f = scorer.score("Hi")
        _, long_f = scorer.score("Why is the API returning a 500 error when I make a POST request?")
        assert long_f.transcript_length > short_f.transcript_length


class TestRecentInteraction:

    def test_recent_interaction_boost(self, scorer):
        s_recent, f = scorer.score("Yes", last_interaction_age_seconds=5.0)
        s_old, _ = scorer.score("Yes", last_interaction_age_seconds=60.0)
        assert f.recent_interaction > 0.0
        assert s_recent > s_old

    def test_no_interaction_history(self, scorer):
        _, features = scorer.score("Hello", last_interaction_age_seconds=None)
        assert features.recent_interaction == 0.0


class TestCustomWeights:

    def test_zero_weights_zero_score(self):
        scorer = AttentionScorer(weights=ScorerWeights(
            explicit_address=0, question=0, request_language=0,
            recent_interaction=0, technical_terms=0, transcript_length=0,
            background_signal=0,
        ))
        salience, _ = scorer.score("Why is the API failing?")
        assert salience == 0.0

    def test_heavy_question_weight(self):
        scorer = AttentionScorer(weights=ScorerWeights(question=1.0))
        salience, _ = scorer.score("What time is it?")
        assert salience > 0.3
