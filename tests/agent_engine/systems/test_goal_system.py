# tests/systems/test_goal_system.py

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from agent_core.core.ecs.component import (
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
)

# Subject under test
from agent_engine.systems.goal_system import GoalSystem

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
    # CORRECTED: Added a 6th memory to ensure the check passes comfortably.
    mem_comp.episodic_memory = [
        {
            "outcome": 0.8,
            "action": {"action_type": "EXTRACT"},
            "outcome_details": {"status": "SUCCESS"},
        },
        {
            "outcome": 0.9,
            "action": {"action_type": "EXTRACT"},
            "outcome_details": {"status": "SUCCESS"},
        },
        {
            "outcome": 0.7,
            "action": {"action_type": "EXTRACT"},
            "outcome_details": {"status": "SUCCESS"},
        },
        {
            "outcome": 0.8,
            "action": {"action_type": "EXTRACT"},
            "outcome_details": {"status": "SUCCESS"},
        },
        {
            "outcome": 0.9,
            "action": {"action_type": "EXTRACT"},
            "outcome_details": {"status": "SUCCESS"},
        },
        {
            "outcome": 0.9,
            "action": {"action_type": "EXTRACT"},
            "outcome_details": {"status": "SUCCESS"},
        },
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
    # Note: We capture the patch object as mock_kmeans
    with (
        patch("agent_engine.systems.goal_system.get_embedding_with_cache"),
        patch("agent_engine.systems.goal_system.KMeans") as mock_kmeans,
    ):
        # 1. Create a mock for the KMeans instance that is returned by fit()
        mock_kmeans_instance = MagicMock()
        # 2. Set the attributes that the application code will access
        mock_kmeans_instance.n_clusters = 1
        # The number of labels must match the number of memories (6)
        mock_kmeans_instance.labels_ = np.zeros(6, dtype=int)

        # 3. Configure the mock KMeans class to return our instance when fit() is called
        mock_kmeans.return_value.fit.return_value = mock_kmeans_instance

        # --- The rest of the fixture is unchanged ---
        mock_simulation_state.event_bus = mock_event_bus

        mock_config = SimpleNamespace(
            agent=SimpleNamespace(cognitive=SimpleNamespace(embeddings=SimpleNamespace(main_embedding_dim=4))),
            llm=SimpleNamespace(embedding_model="test-model"),
        )
        system = GoalSystem(
            simulation_state=mock_simulation_state,
            config=mock_config,
            cognitive_scaffold=mock_cognitive_scaffold,
        )
        yield system


# --- Test Cases ---


def test_on_update_goals_event(goal_system, mock_simulation_state, mock_cognitive_scaffold, mocker):
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

    # CORRECTED: We mock the embedding function for this specific test
    # to ensure the new goal is scored higher than the existing one.
    def embedding_side_effect(text, *args, **kwargs):
        # If we are embedding the new goal name, return a specific vector
        if "acquire resources" in text:
            return np.array([1.0, 0.0, 0.0, 0.0])
        # If we are embedding the context, return a very similar vector
        elif "My situation" in text:
            return np.array([0.9, 0.1, 0.0, 0.0])
        # For anything else (like the old goal), return a different vector
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])

    # Patch the function where it is used inside the goal_system module
    mocker.patch(
        "agent_engine.systems.goal_system.get_embedding_with_cache",
        side_effect=embedding_side_effect,
    )

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
    event_data = {
        "entity_id": "agent_missing_comps",
        "narrative": "...",
        "current_tick": 100,
    }
    mock_simulation_state.entities["agent_missing_comps"] = {}  # No components

    # Act
    goal_system.on_update_goals_event(event_data)

    # Assert
    # The system should exit early without calling the scaffold
    mock_cognitive_scaffold.query.assert_not_called()
