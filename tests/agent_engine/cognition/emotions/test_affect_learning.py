# agent-engine/tests/cognition/emotions/test_affect_learning.py

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from agent_engine.cognition.emotions.affect_base import AffectiveExperience

# Subject under test
from agent_engine.cognition.emotions.affect_learning import (
    discover_emotions,
    name_experience_cluster,
)

# --- Fixtures ---


@pytest.fixture
def mock_cognitive_scaffold():
    """Mocks the CognitiveScaffold. Its query() method returns a string."""
    scaffold = MagicMock()
    # CORRECTED: The query() method returns a string, not a tuple.
    scaffold.query.side_effect = ["joy", "frustration", "calm"]
    return scaffold


@pytest.fixture
def mock_action_registry(mocker):
    """Mocks the global action_registry."""
    registry = MagicMock()
    registry.action_ids = ["action_a", "action_b"]
    mock_action_a_class = MagicMock()
    mock_action_a_instance = MagicMock()
    mock_action_a_instance.name = "Action A"
    mock_action_a_class.return_value = mock_action_a_instance
    mock_action_b_class = MagicMock()
    mock_action_b_instance = MagicMock()
    mock_action_b_instance.name = "Action B"
    mock_action_b_class.return_value = mock_action_b_instance
    registry.get_action.side_effect = lambda action_id: {
        "action_a": mock_action_a_class,
        "action_b": mock_action_b_class,
    }[action_id]
    return mocker.patch("agent_engine.cognition.emotions.affect_learning.action_registry", registry)


@pytest.fixture
def mock_affect_component():
    """Creates a mock affect component with a buffer of experiences."""
    affect_comp = MagicMock()
    experiences = [
        AffectiveExperience(
            valence=0.8,
            arousal=0.7,
            prediction_delta_magnitude=0.1,
            predictive_delta_smooth=0.2,
            health_norm=0.9,
            time_norm=0.8,
            res_norm=0.9,
            action_type_one_hot=np.array([1, 0], dtype=np.float32),
            outcome_reward=10,
            prediction_error=2,
            is_positive_outcome=True,
        )
        for _ in range(20)
    ] + [
        AffectiveExperience(
            valence=-0.6,
            arousal=0.8,
            prediction_delta_magnitude=0.9,
            predictive_delta_smooth=0.7,
            health_norm=0.4,
            time_norm=0.3,
            res_norm=0.2,
            action_type_one_hot=np.array([0, 1], dtype=np.float32),
            outcome_reward=-5,
            prediction_error=-3,
            is_positive_outcome=False,
        )
        for _ in range(20)
    ]
    buffer_mock = MagicMock()
    buffer_mock.__iter__.return_value = iter(experiences)
    buffer_mock.__len__.return_value = len(experiences)
    affect_comp.affective_experience_buffer = buffer_mock
    affect_comp.learned_emotion_clusters = {}
    return affect_comp


@pytest.fixture
def default_config():
    """CORRECTED: Provides a mock config object with the required nested structure."""
    return SimpleNamespace(learning=SimpleNamespace(memory=SimpleNamespace(emotion_cluster_min_data=10)))


# --- Test Cases ---


def test_name_experience_cluster(mock_cognitive_scaffold, mock_action_registry):
    """
    Tests that the LLM prompt is constructed correctly and the response is parsed.
    """
    # Arrange
    experiences = [
        AffectiveExperience(
            valence=0.8,
            arousal=0.7,
            prediction_delta_magnitude=0.1,
            predictive_delta_smooth=0.2,
            health_norm=0.9,
            time_norm=0.8,
            res_norm=0.9,
            action_type_one_hot=np.array([1, 0]),
            outcome_reward=10,
            prediction_error=2,
            is_positive_outcome=True,
        )
    ]
    prompt_template = "Summaries: {summaries}. Name this."

    # Act
    name = name_experience_cluster(experiences, mock_cognitive_scaffold, "agent1", 100, prompt_template, "test_purpose")

    # Assert
    assert name == "joy"
    mock_cognitive_scaffold.query.assert_called_once()
    actual_prompt = mock_cognitive_scaffold.query.call_args[1]["prompt"]
    assert "Summaries:" in actual_prompt
    assert "Action: Action A" in actual_prompt


@patch("agent_engine.cognition.emotions.affect_learning.KMeans")
def test_discover_emotions_success(
    mock_kmeans, mock_affect_component, mock_cognitive_scaffold, default_config, mock_action_registry
):
    """
    Tests the successful discovery and naming of emotion clusters.
    """
    # Arrange
    mock_kmeans_instance = MagicMock()
    mock_kmeans_instance.labels_ = np.array([0] * 20 + [1] * 20)
    mock_kmeans_instance.cluster_centers_ = np.array([[0.5] * 12, [-0.5] * 12])
    mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

    # Act
    discover_emotions(mock_affect_component, mock_cognitive_scaffold, "agent1", 100, default_config)

    # Assert
    assert len(mock_affect_component.learned_emotion_clusters) == 2
    assert "joy" in mock_affect_component.learned_emotion_clusters
    assert "frustration" in mock_affect_component.learned_emotion_clusters
    mock_affect_component.affective_experience_buffer.clear.assert_called_once()


def test_discover_emotions_insufficient_data(mock_affect_component, mock_cognitive_scaffold, default_config):
    """
    Tests that the function returns early if there's not enough data.
    """
    # Arrange
    mock_affect_component.affective_experience_buffer.__len__.return_value = 5

    # Act
    discover_emotions(mock_affect_component, mock_cognitive_scaffold, "agent1", 100, default_config)

    # Assert
    assert not mock_affect_component.learned_emotion_clusters
    mock_cognitive_scaffold.query.assert_not_called()


@patch("agent_engine.cognition.emotions.affect_learning._cluster_experiences", return_value=(None, None))
def test_discover_emotions_clustering_fails(
    mock_cluster_exp, mock_affect_component, mock_cognitive_scaffold, default_config
):
    """
    Tests that the function handles failures from the clustering step gracefully.
    """
    # Act
    discover_emotions(mock_affect_component, mock_cognitive_scaffold, "agent1", 100, default_config)

    # Assert
    assert not mock_affect_component.learned_emotion_clusters
    mock_cognitive_scaffold.query.assert_not_called()
