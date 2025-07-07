# tests/systems/test_goal_system.py

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import os

# Subject under test
from agent_engine.systems.goal_system import GoalSystem
from agent_core.core.ecs.component import (
    GoalComponent,
    MemoryComponent,
    IdentityComponent,
    EmotionComponent,
)

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # --- Mock Components ---
    goal_comp = GoalComponent(embedding_dim=4)
    goal_comp.symbolic_goals_data = {
        "existing_goal": {
            "embedding": np.array([0.1, 0.2, 0.3, 0.4]),
            "success_history": [1.0, 1.0],
            "last_updated_tick": 10,
        }
    }

    mem_comp = MemoryComponent()
    mem_comp.episodic_memory = [
        {"outcome": 0.8, "action": {"action_type": "EXTRACT"}, "outcome_details": {"status": "SUCCESS"}},
        {"outcome": 0.9, "action": {"action_type": "EXTRACT"}, "outcome_details": {"status": "SUCCESS"}},
        {"outcome": 0.7, "action": {"action_type": "EXTRACT"}, "outcome_details": {"status": "SUCCESS"}},
        {"outcome": 0.8, "action": {"action_type": "EXTRACT"}, "outcome_details": {"status": "SUCCESS"}},
        {"outcome": 0.9, "action": {"action_type": "EXTRACT"}, "outcome_details": {"status": "SUCCESS"}},
    ]

    id_comp = MagicMock(spec=IdentityComponent)
    id_comp.embedding = np.array([0.5, 0.5, 0.5, 0.5])
    id_comp.salient_traits_cache = {"persistent": 0.8}

    emotion_comp = MagicMock(spec=EmotionComponent)
    emotion_comp.current_emotion_category = "determined"

    state.entities = {
        "agent1": {
            GoalComponent: goal_comp,
            MemoryComponent: mem_comp,
            IdentityComponent: id_comp,
            EmotionComponent: emotion_comp,
        }
    }
    return state


@pytest.fixture
def mock_cognitive_scaffold():
    """Mocks the CognitiveScaffold to return predictable goal names."""
    scaffold = MagicMock()
    scaffold.query.return_value = "acquire resources"  # Lowercase to match code
    return scaffold


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
def mock_openai_key(mocker):
    """Mocks the OpenAI API key environment variable."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})


@pytest.fixture
def goal_system(
    mock_simulation_state,
    mock_cognitive_scaffold,
    mock_event_bus,
    mock_openai_key,
):
    """Provides an initialized GoalSystem with all dependencies mocked."""
    # Mock all the OpenAI and KMeans functionality to prevent network calls
    with (
        patch("agent_engine.systems.goal_system.get_embedding_with_cache") as mock_get_embedding,
        patch("agent_engine.systems.goal_system.KMeans") as mock_kmeans,
    ):
        # Configure the embedding mock to return predictable vectors
        mock_get_embedding.return_value = np.random.rand(4).astype(np.float32)

        # Mock the KMeans clustering algorithm
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.n_clusters = 1
        mock_kmeans_instance.labels_ = np.zeros(5)  # All memories in one cluster
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        # The system uses the event bus from the simulation state
        mock_simulation_state.event_bus = mock_event_bus

        system = GoalSystem(
            simulation_state=mock_simulation_state,
            config={"goal_invention_min_successes": 5, "llm": {}},
            cognitive_scaffold=mock_cognitive_scaffold,
        )
        yield system


# --- Test Cases ---


def test_on_update_goals_event(goal_system, mock_simulation_state, mock_cognitive_scaffold):
    """
    Tests the main event handler to ensure it orchestrates the goal update cycle.
    """
    # Arrange
    event_data = {
        "entity_id": "agent1",
        "narrative": "I have been successfully gathering resources.",
        "current_tick": 100,
    }
    goal_comp = mock_simulation_state.entities["agent1"][GoalComponent]

    # Act
    goal_system.on_update_goals_event(event_data)

    # Assert
    # 1. Check that a new goal was invented via the LLM
    mock_cognitive_scaffold.query.assert_called_once()
    assert "acquire resources" in goal_comp.symbolic_goals_data

    # 2. Check that the new goal was selected as the current best goal
    assert goal_comp.current_symbolic_goal == "acquire resources"


def test_invent_and_refine_goals_insufficient_memory(goal_system, mock_simulation_state, mock_cognitive_scaffold):
    """
    Tests that goal invention is skipped if there are not enough successful memories.
    """
    # Arrange
    mem_comp = mock_simulation_state.entities["agent1"][MemoryComponent]
    mem_comp.episodic_memory = []  # Clear memories

    # Act
    goal_system._invent_and_refine_goals("agent1", mock_simulation_state.entities["agent1"], 100)

    # Assert
    mock_cognitive_scaffold.query.assert_not_called()


def test_select_best_goal(goal_system, mock_simulation_state):
    """
    Tests the goal scoring and selection logic.
    """
    # Arrange
    components = mock_simulation_state.entities["agent1"]
    goal_comp = components[GoalComponent]

    # Add another goal that is less aligned with the context
    goal_comp.symbolic_goals_data["less_relevant_goal"] = {
        "embedding": np.array([-1.0, -1.0, -1.0, -1.0]),  # Opposite of context
        "success_history": [0.1],  # Low success
        "last_updated_tick": 5,
    }

    narrative = "My goal is to achieve the existing goal."

    # Mock the embedding function to return a specific context embedding
    with patch("agent_engine.systems.goal_system.get_embedding_with_cache") as mock_embed:
        # Return an embedding similar to the "existing_goal"
        mock_embed.return_value = np.array([0.1, 0.2, 0.3, 0.4])

        # Act
        best_goal = goal_system._select_best_goal("agent1", components, narrative)

    # Assert
    # The "existing_goal" should be selected because its embedding will be more
    # similar to the context narrative's embedding.
    assert best_goal == "existing_goal"


def test_on_update_goals_event_missing_components(goal_system, mock_simulation_state, mock_cognitive_scaffold):
    """
    Tests that the system handles an entity with missing components gracefully.
    """
    # Arrange
    event_data = {"entity_id": "agent_missing_comps", "narrative": "...", "current_tick": 100}
    mock_simulation_state.entities["agent_missing_comps"] = {}  # No components

    # Act
    goal_system.on_update_goals_event(event_data)

    # Assert
    # The system should exit early without calling the scaffold
    mock_cognitive_scaffold.query.assert_not_called()
