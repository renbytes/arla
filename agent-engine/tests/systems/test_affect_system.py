# tests/systems/test_affect_system.py

from unittest.mock import MagicMock, patch

import pytest
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    AffectComponent,
    EmotionComponent,
    GoalComponent,
)

# Subject under test
from agent_engine.systems.affect_system import AffectSystem

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # Mock components
    affect_comp = AffectComponent(affective_buffer_maxlen=100)
    affect_comp.predictive_delta_smooth = 0.5
    affect_comp.prev_reward = 0.0  # Set the dynamic attribute for the test
    affect_comp.learned_emotion_clusters = {}

    emotion_comp = EmotionComponent(valence=0.1, arousal=0.2)
    goal_comp = GoalComponent(embedding_dim=4)
    goal_comp.current_symbolic_goal = "test_goal"

    state.entities = {
        "agent1": {
            AffectComponent: affect_comp,
            EmotionComponent: emotion_comp,
            GoalComponent: goal_comp,
        }
    }
    # Mock the get_entities_with_components method
    state.get_entities_with_components.return_value = {"agent1": state.entities["agent1"]}
    return state


@pytest.fixture
def mock_providers():
    """Mocks the vitality and controllability provider interfaces."""
    vitality_provider = MagicMock()
    vitality_provider.get_normalized_vitality_metrics.return_value = {
        "health_norm": 0.8,
        "time_norm": 0.7,
        "resources_norm": 0.6,
    }

    controllability_provider = MagicMock()
    controllability_provider.get_controllability_score.return_value = 0.75

    return {
        "vitality": vitality_provider,
        "controllability": controllability_provider,
    }


@pytest.fixture
def mock_emotional_dynamics(mocker):
    """Mocks the EmotionalDynamics class."""
    mock_dynamics_class = mocker.patch("agent_engine.systems.affect_system.EmotionalDynamics")
    mock_instance = mock_dynamics_class.return_value
    mock_instance.update_emotion_with_appraisal.return_value = {
        "valence": 0.5,
        "arousal": 0.6,
    }
    return mock_instance


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
@patch("agent_engine.systems.affect_system.action_registry")
def affect_system(
    mock_registry,
    mock_simulation_state,
    mock_providers,
    mock_emotional_dynamics,
    mock_event_bus,
):
    """Provides an initialized AffectSystem with its core dependencies mocked."""
    mock_registry.action_ids = ["test_action"]
    mock_simulation_state.event_bus = mock_event_bus

    system = AffectSystem(
        simulation_state=mock_simulation_state,
        config={"learning": {"memory": {"emotion_cluster_min_data": 50}}},
        cognitive_scaffold=MagicMock(),
        vitality_metrics_provider=mock_providers["vitality"],
        controllability_provider=mock_providers["controllability"],
    )
    return system


# --- Test Cases ---


@patch(
    "agent_engine.systems.affect_system.get_emotion_from_affect",
    return_value="discovered_joy",
)
def test_on_action_executed_updates_affect_and_emotion(
    mock_get_emotion,
    affect_system,
    mock_simulation_state,
    mock_providers,
    mock_emotional_dynamics,
):
    """
    Tests the full event handler cycle for a standard action outcome.
    """
    # Arrange
    action_outcome = ActionOutcome(success=True, message="", base_reward=5.0, details={})
    mock_action_plan = MagicMock()
    mock_action_plan.action_type.action_id = "test_action"

    event_data = {
        "entity_id": "agent1",
        "action_outcome": action_outcome,
        "action_plan": mock_action_plan,
        "current_tick": 100,
    }

    affect_comp = mock_simulation_state.entities["agent1"][AffectComponent]
    emotion_comp = mock_simulation_state.entities["agent1"][EmotionComponent]

    # Act
    affect_system.on_action_executed(event_data)

    # Assert
    # 1. Verify providers were called
    mock_providers["vitality"].get_normalized_vitality_metrics.assert_called_once()
    mock_providers["controllability"].get_controllability_score.assert_called_once()

    # 2. Verify the emotional dynamics model was updated
    mock_emotional_dynamics.update_emotion_with_appraisal.assert_called_once()

    # 3. Verify the agent's EmotionComponent was updated with the new state
    assert emotion_comp.valence == 0.5
    assert emotion_comp.arousal == 0.6

    # 4. Verify a new AffectiveExperience was added to the buffer
    assert len(affect_comp.affective_experience_buffer) == 1

    # 5. Verify the new emotion category was set
    assert emotion_comp.current_emotion_category == "discovered_joy"


def test_on_action_executed_missing_components(affect_system, mock_simulation_state, mock_providers):
    """
    Tests that the system gracefully handles an entity with missing components.
    """
    # Arrange
    event_data = {
        "entity_id": "agent_missing_comps",
        "action_outcome": MagicMock(),
        "action_plan": MagicMock(),
        "current_tick": 100,
    }
    mock_simulation_state.entities["agent_missing_comps"] = {}  # No components

    # Act
    affect_system.on_action_executed(event_data)

    # Assert
    # None of the providers or models should have been called
    mock_providers["vitality"].get_normalized_vitality_metrics.assert_not_called()
    mock_providers["controllability"].get_controllability_score.assert_not_called()


@patch("agent_engine.systems.affect_system.discover_emotions")
def test_emotion_discovery_is_triggered(mock_discover, affect_system, mock_simulation_state):
    """
    Tests that the discover_emotions function is called when the buffer is full.
    """
    # Arrange
    # FIX: Set a value that won't cause division by zero in the application code.
    affect_system.config["learning"]["memory"]["emotion_cluster_min_data"] = 4
    event_data = {
        "entity_id": "agent1",
        "action_outcome": ActionOutcome(success=True, message="", base_reward=5.0, details={}),
        "action_plan": MagicMock(),
        "current_tick": 100,
    }
    event_data["action_plan"].action_type.action_id = "test_action"

    # Act
    # We need to call the event handler enough times to fill the buffer
    for _ in range(4):
        affect_system.on_action_executed(event_data)

    # Assert
    mock_discover.assert_called_once()


@pytest.mark.asyncio
async def test_passive_dissonance_decay(affect_system, mock_simulation_state):
    """
    Tests the passive update method that decays cognitive dissonance.
    """
    # Arrange
    affect_comp = mock_simulation_state.entities["agent1"][AffectComponent]
    affect_comp.cognitive_dissonance = 0.5

    # Act
    # FIX: Use a tick value that will pass the `(current_tick + 1) % 10 == 0` check.
    await affect_system.update(current_tick=9)  # Passive update

    # Assert
    assert affect_comp.cognitive_dissonance < 0.5
    assert affect_comp.cognitive_dissonance == pytest.approx(0.5 * 0.99)
