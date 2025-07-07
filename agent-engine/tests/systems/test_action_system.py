# tests/systems/test_action_system.py

from unittest.mock import MagicMock
import pytest

# Subject under test
from agent_engine.systems.action_system import ActionSystem
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionOutcomeComponent,
    ActionPlanComponent,
    CompetenceComponent,
)

# --- Fixtures ---


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # Mock components that the system will update
    state.entities = {
        "agent1": {
            ActionOutcomeComponent: ActionOutcomeComponent(),
            CompetenceComponent: CompetenceComponent(),
        }
    }

    # FIX: Mock the get_component method to return the correct component instance.
    def get_component_side_effect(entity_id, component_type):
        return state.entities.get(entity_id, {}).get(component_type)

    state.get_component.side_effect = get_component_side_effect

    return state


@pytest.fixture
def mock_reward_calculator():
    """Mocks the RewardCalculatorInterface."""
    calculator = MagicMock()
    # Simulate the calculator returning a final reward and a breakdown dictionary
    calculator.calculate_final_reward.return_value = (15.0, {"base": 10.0, "bonus": 5.0})
    return calculator


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
def action_system(mock_simulation_state, mock_reward_calculator, mock_event_bus):
    """Provides an initialized ActionSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus

    system = ActionSystem(
        simulation_state=mock_simulation_state,
        config={},
        cognitive_scaffold=MagicMock(),
        reward_calculator=mock_reward_calculator,
    )
    return system


# --- Test Cases ---


def test_on_action_chosen_dispatches_specific_event(action_system, mock_event_bus):
    """
    Tests that receiving an 'action_chosen' event correctly publishes a
    more specific 'execute_{action_id}_action' event.
    """
    # Arrange
    # FIX: Use spec=ActionInterface to ensure isinstance checks in the system pass.
    mock_action_type = MagicMock(spec=ActionInterface)
    mock_action_type.action_id = "test_action"

    action_plan = ActionPlanComponent(action_type=mock_action_type)
    event_data = {"action_plan_component": action_plan}

    # Act
    action_system.on_action_chosen(event_data)

    # Assert
    mock_event_bus.publish.assert_called_once_with("execute_test_action_action", event_data)


def test_on_action_outcome_ready_calculates_reward_and_publishes(
    action_system, mock_simulation_state, mock_reward_calculator, mock_event_bus
):
    """
    Tests the main logic: processing an outcome, using the reward calculator,
    updating components, and publishing the final 'action_executed' event.
    """
    # Arrange
    base_outcome = ActionOutcome(success=True, message="", base_reward=10.0, details={"target": "rock"})

    mock_action_type = MagicMock(spec=ActionInterface)
    mock_action_type.name = "Test Action"
    mock_action_type.action_id = "test_action_id"
    mock_action_plan = ActionPlanComponent(action_type=mock_action_type, intent=MagicMock(name="SOLITARY"))

    event_data = {
        "entity_id": "agent1",
        "action_outcome": base_outcome,
        "original_action_plan": mock_action_plan,
        "current_tick": 50,
    }

    aoc = mock_simulation_state.entities["agent1"][ActionOutcomeComponent]
    cc = mock_simulation_state.entities["agent1"][CompetenceComponent]

    # Act
    action_system.on_action_outcome_ready(event_data)

    # Assert
    # 1. Verify the reward calculator was called with the correct data
    mock_reward_calculator.calculate_final_reward.assert_called_once()

    # 2. Verify the agent's components were updated
    assert aoc.success is True
    assert aoc.reward == 15.0  # The final, subjective reward
    assert aoc.details["reward_breakdown"]["bonus"] == 5.0
    assert cc.action_counts[mock_action_plan.action_type.action_id] == 1

    # 3. Verify the final 'action_executed' event was published
    mock_event_bus.publish.assert_called_once()
    final_event_name, final_event_data = mock_event_bus.publish.call_args[0]

    assert final_event_name == "action_executed"
    assert final_event_data["entity_id"] == "agent1"
    assert final_event_data["action_outcome"].reward == 15.0


async def test_update_method_is_empty(action_system, mock_simulation_state, mock_event_bus):
    """
    Confirms that the system's passive update method does nothing, as it is purely event-driven.
    """
    # Arrange
    # FIX: Add mock_simulation_state to the function signature to make it available.
    initial_state = mock_simulation_state.entities["agent1"][ActionOutcomeComponent].reward

    # Act
    await action_system.update(current_tick=100)

    # Assert
    # The state should be unchanged by the update call
    assert mock_simulation_state.entities["agent1"][ActionOutcomeComponent].reward == initial_state
    mock_event_bus.publish.assert_not_called()


def test_on_action_chosen_handles_invalid_plan(action_system, mock_event_bus):
    """
    Tests that the system gracefully ignores an event with an invalid or missing action plan.
    """
    # Arrange
    event_data = {"action_plan_component": None}

    # Act
    action_system.on_action_chosen(event_data)

    # Assert
    mock_event_bus.publish.assert_not_called()
