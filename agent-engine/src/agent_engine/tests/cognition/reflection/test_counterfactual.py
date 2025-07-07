# tests/cognition/reflection/test_counterfactual.py

import pytest
from unittest.mock import MagicMock

# Subject under test
from agent_engine.cognition.reflection.counterfactual import generate_counterfactual, CounterfactualEpisode
from agent_engine.cognition.reflection.episode import Episode


# --- Fixtures ---


@pytest.fixture
def mock_cognitive_scaffold():
    """Mocks the CognitiveScaffold to return a predictable structured response."""
    scaffold = MagicMock()
    llm_response = """
    ALTERNATIVE ACTION: Negotiate Peace
    PREDICTED OUTCOME: The agent would have avoided conflict and potentially formed an alliance.
    """
    scaffold.query.return_value = llm_response
    return scaffold


@pytest.fixture
def sample_episode():
    """Provides a sample Episode with a key event for counterfactual generation."""
    # This is the most significant event and should be chosen by the function
    key_event = {
        "tick": 50,
        "action_type": "COMBAT",
        "status": "VICTORY",
        "reward": 10.0,
        "action": {"type": "COMBAT", "target": "enemy_1"},
    }
    other_event = {"tick": 48, "action_type": "MOVE", "status": "SUCCESS", "reward": 0.1}

    episode = Episode(
        start_tick=45,
        end_tick=55,
        theme="A Great Battle",
        events=[other_event, key_event],
    )
    return episode


# --- Test Cases ---


def test_generate_counterfactual_success(sample_episode, mock_cognitive_scaffold):
    """
    Tests the successful generation and parsing of a counterfactual episode.
    """
    # Act
    counterfactuals = generate_counterfactual(
        episode=sample_episode,
        cognitive_scaffold=mock_cognitive_scaffold,
        agent_id="agent1",
        current_tick=100,
    )

    # Assert
    # 1. Check that the LLM was queried
    mock_cognitive_scaffold.query.assert_called_once()

    # 2. Check the prompt sent to the LLM contains the key event details
    prompt = mock_cognitive_scaffold.query.call_args[1]["prompt"]
    assert "tick 50" in prompt
    assert "'COMBAT'" in prompt
    assert "'VICTORY'" in prompt
    assert "reward of 10.00" in prompt

    # 3. Check the returned counterfactual object
    assert len(counterfactuals) == 1
    cf_episode = counterfactuals[0]
    assert isinstance(cf_episode, CounterfactualEpisode)
    assert cf_episode.original_episode_theme == "A Great Battle"
    assert cf_episode.counterfactual_action == "Negotiate Peace"
    assert "avoided conflict" in cf_episode.predicted_outcome


def test_generate_counterfactual_empty_episode(mock_cognitive_scaffold):
    """
    Tests that the function returns an empty list for an episode with no events.
    """
    # Arrange
    empty_episode = Episode(start_tick=1, end_tick=2, theme="Nothing happened", events=[])

    # Act
    counterfactuals = generate_counterfactual(
        episode=empty_episode,
        cognitive_scaffold=mock_cognitive_scaffold,
        agent_id="agent1",
        current_tick=100,
    )

    # Assert
    assert counterfactuals == []
    mock_cognitive_scaffold.query.assert_not_called()


def test_generate_counterfactual_llm_failure(sample_episode, mock_cognitive_scaffold):
    """
    Tests that the function handles an exception from the LLM query gracefully.
    """
    # Arrange
    mock_cognitive_scaffold.query.side_effect = Exception("LLM API is down")

    # Act
    counterfactuals = generate_counterfactual(
        episode=sample_episode,
        cognitive_scaffold=mock_cognitive_scaffold,
        agent_id="agent1",
        current_tick=100,
    )

    # Assert
    assert counterfactuals == []
