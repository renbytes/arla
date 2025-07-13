# tests/cognition/reflection/test_validation.py - Fixed version

import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from agent_engine.cognition.reflection.episode import Episode

# Subject under test
from agent_engine.cognition.reflection.validation import (
    RuleValidator,
    calculate_confidence_score,
)

# --- Fixtures ---


@pytest.fixture
def mock_cognitive_scaffold():
    """Mocks the CognitiveScaffold to return a predictable summary."""
    scaffold = MagicMock()
    scaffold.query.return_value = "The agent moved and then successfully extracted resources."
    return scaffold


@pytest.fixture
def sample_episode():
    """Provides a sample Episode for validation context."""
    events = [
        {"tick": 10, "action_type": "MOVE", "reward": 0.1},
        {"tick": 12, "action_type": "EXTRACT", "reward": 5.0},
    ]
    return Episode(start_tick=10, end_tick=15, theme="A successful outing", events=events)


@pytest.fixture
def mock_openai_key(mocker):
    """Mocks the OpenAI API key environment variable."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})


@pytest.fixture
def rule_validator(sample_episode, mock_cognitive_scaffold, mock_openai_key):
    """Provides an initialized RuleValidator with mocked dependencies."""

    # Mock the embedding function to prevent network calls
    with patch("agent_engine.cognition.reflection.validation.get_embedding_with_cache") as mock_get_embedding:
        # Configure the mock embedding function to return different vectors for different inputs
        def embedding_side_effect(text, *args, **kwargs):
            if "happy" in text.lower():
                return np.array([1.0, 0.9, 0.8, 0.7])  # Reflection embedding
            else:
                return np.array([1.0, 0.88, 0.82, 0.71])  # Factual summary embedding (very similar)

        mock_get_embedding.side_effect = embedding_side_effect

        mock_config = SimpleNamespace(
            agent=SimpleNamespace(cognitive=SimpleNamespace(embeddings=SimpleNamespace(main_embedding_dim=4))),
            llm=SimpleNamespace(),  # Add llm attribute for the second access
        )

        validator = RuleValidator(
            episode=sample_episode,
            config=mock_config,
            cognitive_scaffold=mock_cognitive_scaffold,
            agent_id="agent1",
            current_tick=100,
        )
        yield validator


# --- Test Cases for RuleValidator ---


def test_check_coherence():
    """
    Tests the simple logical coherence check.
    """
    # RuleValidator is stateless for this method, so we can instantiate it directly
    validator = RuleValidator(MagicMock(), {}, MagicMock(), "", 0)
    assert validator.check_coherence("This was a good day.") is True
    assert validator.check_coherence("I felt happy and I felt sad.") is False


def test_check_factual_alignment_high_similarity(rule_validator):
    """
    Tests that factual alignment returns a high score for semantically similar texts.
    """
    # The mock embeddings are configured to be very similar
    inference = "I felt happy about my success."
    alignment_score = rule_validator.check_factual_alignment(inference)

    # Expect a score close to 1.0 due to high cosine similarity
    assert alignment_score > 0.95


def test_check_factual_alignment_embedding_failure(rule_validator):
    """
    Tests that the function handles failures from the embedding service gracefully.
    """
    # Mock the embedding function to return None (simulating failure)
    with patch(
        "agent_engine.cognition.reflection.validation.get_embedding_with_cache",
        return_value=None,
    ):
        alignment_score = rule_validator.check_factual_alignment("An inference.")
        assert alignment_score == 0.0


def test_check_factual_alignment_empty_inference(rule_validator):
    """
    Tests that an empty inference string results in a zero alignment score.
    """
    alignment_score = rule_validator.check_factual_alignment("")
    assert alignment_score == 0.0


# --- Test Cases for calculate_confidence_score ---


def test_calculate_confidence_score_coherent_and_aligned():
    """
    Tests the confidence calculation for a coherent and factually aligned inference.
    """
    # Arrange
    coherence = True
    factual_alignment = 0.9
    token_uncertainty = 0.8

    # Act
    confidence = calculate_confidence_score(coherence, factual_alignment, token_uncertainty)

    # Assert
    expected_confidence = (0.9 * 0.7) + (0.8 * 0.3)  # 0.63 + 0.24 = 0.87
    assert math.isclose(confidence, expected_confidence, rel_tol=1e-2)


def test_calculate_confidence_score_incoherent():
    """
    Tests that an incoherent inference results in a confidence score of 0.0,
    regardless of other factors.
    """
    # Arrange
    coherence = False
    factual_alignment = 0.9

    # Act
    confidence = calculate_confidence_score(coherence, factual_alignment)

    # Assert
    assert confidence == 0.0
