# tests/systems/test_affect_system.py

import sys
from unittest.mock import MagicMock

import pytest

# Mock non-essential modules before other imports
sys.modules["sqlalchemy.ext.asyncio.engine"] = MagicMock()
sys.modules["sqlalchemy.ext.asyncio"] = MagicMock()

# Imports from the core library are now valid because of py.typed
from agent_core.agents.actions.action_interface import ActionInterface  # noqa: E402
from agent_core.agents.actions.base_action import ActionOutcome, Intent  # noqa: E402
from agent_core.core.ecs.component import (  # noqa: E402
    ActionPlanComponent,
    AffectComponent,
    EmotionComponent,
    GoalComponent,
)

# These components are NOT in agent_core, so we mock them for testing the system
from unittest.mock import MagicMock as MockHealthComponent  # noqa: E402
from unittest.mock import MagicMock as MockInventoryComponent  # noqa: E402
from unittest.mock import MagicMock as MockFailedStatesComponent  # noqa: E402
from unittest.mock import MagicMock as MockPositionComponent  # noqa: E402
from unittest.mock import MagicMock as MockTimeBudgetComponent  # noqa: E402

# Imports from the engine library being tested
from agent_engine.systems.affect_system import AffectSystem  # noqa: E402
from agent_engine.environment.interface import EnvironmentInterface  # noqa: E402


@pytest.fixture
def mock_config():
    """Provides a default config for tests."""
    return {
        "learning": {"memory": {"emotion_cluster_min_data": 5}},
        "agent": {
            "cognitive": {"architecture_flags": {"enable_emotion_clustering": True}},
            "foundational": {"vitals": {"initial_resources": 100.0}},
        },
        "valence_decay_rate": 0.95,
        "arousal_decay_rate": 0.90,
        "valence_learning_rate": 0.2,
        "arousal_learning_rate": 0.3,
        "emotional_noise_std": 0.0,
    }


@pytest.fixture
def mock_simulation_state():
    """Provides a mock simulation state."""
    sim_state = MagicMock()
    sim_state.event_bus = MagicMock()
    sim_state.entities = {}
    sim_state.environment = MagicMock(spec=EnvironmentInterface)
    sim_state.get_entities_with_components.return_value = sim_state.entities
    return sim_state


@pytest.fixture
def core_agent_components(mock_simulation_state):
    """Provides a dictionary with core and mocked world-specific components."""
    return {
        AffectComponent: AffectComponent(affective_buffer_maxlen=10),
        EmotionComponent: EmotionComponent(valence=0.1, arousal=0.2),
        GoalComponent: GoalComponent(embedding_dim=8),
        # Mock world-specific components
        MockHealthComponent: MockHealthComponent(initial_health=100.0),
        MockTimeBudgetComponent: MockTimeBudgetComponent(initial_time_budget=1000.0),
        MockInventoryComponent: MockInventoryComponent(initial_resources=100.0),
        MockFailedStatesComponent: MockFailedStatesComponent(),
        MockPositionComponent: MockPositionComponent(position=(5, 5)),
    }


@pytest.fixture
def affect_system(mock_simulation_state, mock_config, mocker):
    """Initializes the AffectSystem with mocked dependencies."""
    mocker.patch("agent_engine.systems.affect_system.discover_emotions")
    mocker.patch("agent_engine.systems.affect_system.get_emotion_from_affect", return_value="curiosity")
    mocker.patch("agent_engine.systems.affect_system.get_config_value", return_value=100.0)
    mock_registry = mocker.patch("agent_engine.systems.affect_system.action_registry")
    mock_registry.action_ids = ["test_action"]

    system = AffectSystem(mock_simulation_state, mock_config, MagicMock())
    system.discover_emotions = sys.modules["agent_engine.systems.affect_system.discover_emotions"]
    system.get_emotion_from_affect = sys.modules["agent_engine.systems.affect_system.get_emotion_from_affect"]

    return system


def test_on_action_executed_orchestrates_processing(
    affect_system, mock_simulation_state, core_agent_components, mocker
):
    """Tests that the main event handler correctly fetches dependencies and calls the core logic."""
    entity_id = "agent_1"
    mock_simulation_state.entities = {entity_id: core_agent_components}

    action_outcome = ActionOutcome(success=True, message="Success", base_reward=10.0)
    action_plan = ActionPlanComponent(action_type=mocker.MagicMock(spec=ActionInterface), intent=Intent.SOLITARY)
    event_data = {
        "entity_id": entity_id,
        "action_outcome": action_outcome,
        "action_plan": action_plan,
        "current_tick": 123,
    }

    process_spy = mocker.spy(affect_system, "_process_affective_response")
    affect_system.on_action_executed(event_data)
    process_spy.assert_called_once()
