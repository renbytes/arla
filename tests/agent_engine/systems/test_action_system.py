# agent-engine/tests/systems/test_action_system.py

from unittest.mock import MagicMock

import pytest
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome, Intent
from agent_core.core.ecs.component import (
    ActionOutcomeComponent,
    ActionPlanComponent,
    CompetenceComponent,
    TimeBudgetComponent,
)

# Subject under test
from agent_engine.systems.action_system import ActionSystem

# Fixtures


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState and its contained components."""
    state = MagicMock()

    # Mock components that the system will update
    state.entities = {
        "agent1": {
            ActionOutcomeComponent: ActionOutcomeComponent(),
            CompetenceComponent: CompetenceComponent(),
            TimeBudgetComponent: TimeBudgetComponent(initial_time_budget=100.0, lifespan_std_dev_percent=0.0),
        }
    }

    # Configure the mock's get_component method to return the correct component instance.
    def get_component_side_effect(entity_id, component_type):
        return state.entities.get(entity_id, {}).get(component_type)

    state.get_component.side_effect = get_component_side_effect
    return state


@pytest.fixture
def mock_reward_calculator():
    """Mocks the RewardCalculatorInterface."""
    calculator = MagicMock()
    # Simulate the calculator returning a final reward and a breakdown dictionary
    calculator.calculate_final_reward.return_value = (
        15.0,
        {"base": 10.0, "bonus": 5.0},
    )
    return calculator


@pytest.fixture
def mock_event_bus():
    """Mocks the EventBus."""
    return MagicMock()


@pytest.fixture
def action_system(mock_simulation_state, mock_reward_calculator, mock_event_bus):
    """Provides an initialized ActionSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus

    # Mock the config to have a value for the action cost
    mock_config = {"agent": {"costs": {"actions": {"base": 1.0}}}}

    system = ActionSystem(
        simulation_state=mock_simulation_state,
        config=mock_config,
        cognitive_scaffold=MagicMock(),
        reward_calculator=mock_reward_calculator,
    )
    return system


# Test Cases


class TestActionSystem:
    def test_on_action_chosen_dispatches_specific_event(self, action_system, mock_event_bus):
        """
        Tests that receiving an 'action_chosen' event correctly publishes a
        more specific 'execute_{action_id}_action' event.
        """
        # Arrange
        # Use spec=ActionInterface to ensure isinstance checks in the system pass.
        mock_action_type = MagicMock(spec=ActionInterface)
        mock_action_type.action_id = "test_action"

        action_plan = ActionPlanComponent(action_type=mock_action_type)
        event_data = {"action_plan_component": action_plan}

        # Act
        action_system.on_action_chosen(event_data)

        # Assert
        mock_event_bus.publish.assert_called_once_with("execute_test_action_action", event_data)

    def test_on_action_outcome_ready_full_cycle(
        self,
        action_system,
        mock_simulation_state,
        mock_reward_calculator,
        mock_event_bus,
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
        # Mock the get_base_cost method to return a value
        mock_action_type.get_base_cost.return_value = 1.0

        mock_action_plan = ActionPlanComponent(action_type=mock_action_type, intent=Intent.SOLITARY)

        event_data = {
            "entity_id": "agent1",
            "action_outcome": base_outcome,
            "original_action_plan": mock_action_plan,
            "current_tick": 50,
        }

        aoc = mock_simulation_state.get_component("agent1", ActionOutcomeComponent)
        cc = mock_simulation_state.get_component("agent1", CompetenceComponent)
        tbc = mock_simulation_state.get_component("agent1", TimeBudgetComponent)
        initial_budget = tbc.current_time_budget

        # Act
        action_system.on_action_outcome_ready(event_data)

        # Assert
        # 1. Verify the reward calculator was called correctly
        mock_reward_calculator.calculate_final_reward.assert_called_once()
        # Manually inspect the keyword arguments of the call for robustness
        call_kwargs = mock_reward_calculator.calculate_final_reward.call_args.kwargs
        assert call_kwargs["base_reward"] == 10.0
        assert call_kwargs["action_type"] is mock_action_type
        assert call_kwargs["action_intent"] == "SOLITARY"
        assert call_kwargs["outcome_details"] == {"target": "rock"}
        assert call_kwargs["entity_components"] == mock_simulation_state.entities["agent1"]

        # 2. Verify the agent's components were updated
        assert aoc.success is True
        assert aoc.reward == 15.0  # The final, subjective reward
        assert aoc.details["reward_breakdown"]["bonus"] == 5.0
        assert cc.action_counts[mock_action_plan.action_type.action_id] == 1
        assert tbc.current_time_budget == initial_budget - 1.0  # Check cost deduction

        # 3. Verify the final 'action_executed' event was published correctly
        mock_event_bus.publish.assert_called_once()
        final_event_name, final_event_data = mock_event_bus.publish.call_args[0]

        assert final_event_name == "action_executed"
        assert final_event_data["entity_id"] == "agent1"
        assert final_event_data["action_outcome"].reward == 15.0

    @pytest.mark.asyncio
    async def test_update_method_is_empty(self, action_system):
        """
        Confirms that the system's passive update method does nothing, as it is purely event-driven.
        """
        # This test primarily ensures the async method can be called without error.
        try:
            await action_system.update(current_tick=100)
        except Exception as e:
            pytest.fail(f"ActionSystem.update() raised an unexpected exception: {e}")

    def test_on_action_chosen_handles_invalid_plan(self, action_system, mock_event_bus):
        """
        Tests that the system gracefully ignores an event with an invalid or missing action plan.
        """
        # Case 1: action_plan_component is None
        action_system.on_action_chosen({"action_plan_component": None})
        mock_event_bus.publish.assert_not_called()

        # Case 2: action_type is not an ActionInterface
        invalid_plan = ActionPlanComponent(action_type="not_an_action_object")
        action_system.on_action_chosen({"action_plan_component": invalid_plan})
        mock_event_bus.publish.assert_not_called()

    def test_on_action_outcome_ready_handles_invalid_plan(self, action_system, mock_event_bus):
        """
        Tests that the outcome handler gracefully ignores an event with an invalid action plan.
        """
        # Arrange
        event_data = {
            "entity_id": "agent1",
            "action_outcome": MagicMock(),
            "original_action_plan": ActionPlanComponent(action_type="not_an_action"),  # Invalid type
            "current_tick": 50,
        }

        # Act
        action_system.on_action_outcome_ready(event_data)

        # Assert
        # The final "action_executed" event should not be published
        mock_event_bus.publish.assert_not_called()
