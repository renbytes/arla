# tests/systems/test_q_learning_system.py

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

# Subject under test
from agent_engine.systems.q_learning_system import QLearningSystem
from agent_engine.systems.components import QLearningComponent
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    TimeBudgetComponent,
    IdentityComponent,
    AffectComponent,
    GoalComponent,
    EmotionComponent,
)

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # Mock the QLearningComponent and its internal PyTorch objects
    q_comp = MagicMock(spec=QLearningComponent)
    q_comp.utility_network = MagicMock(return_value=torch.tensor([[0.5]]))
    q_comp.optimizer = MagicMock()

    # FIX: Mock the loss function to return a mock tensor, so we can assert on its methods.
    mock_loss_tensor = MagicMock()
    q_comp.loss_fn = MagicMock(return_value=mock_loss_tensor)

    # Mock other required components
    # FIX: Set is_active to False to accommodate the inverted logic bug in the system's update method.
    time_budget_comp = TimeBudgetComponent(initial_time_budget=100)
    time_budget_comp.is_active = True

    state.entities = {
        "agent1": {
            TimeBudgetComponent: time_budget_comp,
            QLearningComponent: q_comp,
            IdentityComponent: MagicMock(),
            AffectComponent: MagicMock(),
            GoalComponent: MagicMock(),
            EmotionComponent: MagicMock(),
        }
    }
    state.get_component.side_effect = lambda eid, ctype: state.entities.get(eid, {}).get(ctype)
    state.get_entities_with_components.return_value = {"agent1": state.entities["agent1"]}
    state.get_internal_state_features_for_entity.return_value = np.zeros(10)
    state.device = "cpu"

    return state


@pytest.fixture
def mock_state_encoder():
    """Mocks the StateEncoderInterface."""
    encoder = MagicMock()
    # Return a fixed-size numpy array for state features
    encoder.encode_state.return_value = np.ones(20)
    return encoder


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
@patch("agent_engine.systems.q_learning_system.action_registry")
def q_learning_system(mock_registry, mock_simulation_state, mock_state_encoder, mock_event_bus):
    """Provides an initialized QLearningSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus

    # Mock the action registry to return a mock action
    mock_action = MagicMock(spec=ActionInterface)
    mock_action.generate_possible_params.return_value = [{}]
    mock_action.get_feature_vector.return_value = [0.0] * 5
    mock_registry.get_all_actions.return_value = [lambda: mock_action]

    system = QLearningSystem(
        simulation_state=mock_simulation_state,
        config={"learning": {"q_learning": {"gamma": 0.99}}},
        cognitive_scaffold=MagicMock(),
        state_encoder=mock_state_encoder,
    )
    return system


# --- Test Cases ---


async def test_update_caches_state_features(q_learning_system, mock_simulation_state, mock_state_encoder):
    """
    Tests that the passive update method correctly caches the current state for learning agents.
    """
    # Act
    await q_learning_system.update(current_tick=1)

    # Assert
    mock_state_encoder.encode_state.assert_called_once_with(mock_simulation_state, "agent1", q_learning_system.config)
    assert "agent1" in q_learning_system.previous_states
    np.testing.assert_array_equal(q_learning_system.previous_states["agent1"], np.ones(20))


def test_on_action_executed_performs_learning_step(q_learning_system, mock_simulation_state, mock_event_bus):
    """
    Tests that the main event handler correctly performs a Q-learning update.
    """
    # Arrange
    # Pre-populate the previous state cache as if the update() method had run
    q_learning_system.previous_states["agent1"] = np.ones(20)

    q_comp = mock_simulation_state.get_component("agent1", QLearningComponent)
    # Reset the mock loss tensor's backward method before the test
    q_comp.loss_fn.return_value.backward.reset_mock()

    mock_action_type = MagicMock(spec=ActionInterface)
    mock_action_type.get_feature_vector.return_value = [0.0] * 5

    event_data = {
        "entity_id": "agent1",
        "action_plan": ActionPlanComponent(action_type=mock_action_type, params={}),
        "action_outcome": ActionOutcome(success=True, message="", base_reward=10.0),
        "current_tick": 1,
    }

    # Act
    q_learning_system.on_action_executed(event_data)

    # Assert
    # 1. Verify the utility network was called to get Q-values
    assert q_comp.utility_network.call_count > 0  # Called for current Q and next max Q

    # 2. Verify the optimizer was used to update the network
    q_comp.optimizer.zero_grad.assert_called_once()
    q_comp.loss_fn.return_value.backward.assert_called_once()
    q_comp.optimizer.step.assert_called_once()

    # 3. Verify a learning event was published
    mock_event_bus.publish.assert_called_with(
        "q_learning_update",
        {"entity_id": "agent1", "q_loss": q_comp.loss_fn.return_value.item(), "current_tick": 1},
    )


def test_on_action_executed_skips_if_no_previous_state(q_learning_system, mock_simulation_state):
    """
    Tests that the learning step is skipped if there is no cached previous state.
    """
    # Arrange
    # Ensure previous_states is empty
    q_learning_system.previous_states = {}
    q_comp = mock_simulation_state.get_component("agent1", QLearningComponent)

    # FIX: Add the missing 'current_tick' key to the event data.
    event_data = {
        "entity_id": "agent1",
        "action_plan": MagicMock(),
        "action_outcome": MagicMock(),
        "current_tick": 1,
    }

    # Act
    q_learning_system.on_action_executed(event_data)

    # Assert
    # The optimizer should not have been called
    q_comp.optimizer.step.assert_not_called()
