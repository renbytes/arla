# tests/agent_engine/systems/test_action_system.py
import pytest
from unittest.mock import MagicMock, ANY

from agent_core.agents.action_cost_provider_interface import ActionCostProviderInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_outcome import ActionOutcome
from agent_core.core.ecs.component import ActionPlanComponent
from agent_engine.systems.action_system import ActionSystem


@pytest.fixture
def mock_simulation_state():
    """Provides a mock SimulationState object."""
    return MagicMock()


@pytest.fixture
def mock_reward_calculator():
    """Provides a mock RewardCalculatorInterface."""
    mock = MagicMock()
    mock.calculate_final_reward.side_effect = lambda base_reward, **kwargs: (
        base_reward,
        {},
    )
    return mock


@pytest.fixture
def mock_event_bus():
    """Provides a mock EventBus."""
    return MagicMock(name="mock.event_bus")


@pytest.fixture
def mock_action_cost_provider():
    """Provides a mock ActionCostProviderInterface."""
    return MagicMock(spec=ActionCostProviderInterface)


@pytest.fixture
def action_system(
    mock_simulation_state,
    mock_reward_calculator,
    mock_event_bus,
    mock_action_cost_provider,
):
    """Provides an initialized ActionSystem with all dependencies mocked."""
    mock_simulation_state.event_bus = mock_event_bus
    mock_config = {"agent": {"costs": {"actions": {"base": 1.0}}}}

    system = ActionSystem(
        simulation_state=mock_simulation_state,
        config=mock_config,
        cognitive_scaffold=MagicMock(),
        reward_calculator=mock_reward_calculator,
        action_cost_provider=mock_action_cost_provider,
    )
    return system


class TestActionSystem:
    """Test suite for the ActionSystem."""

    def test_on_action_chosen_dispatches_specific_event(
        self, action_system, mock_event_bus
    ):
        """Verify that on_action_chosen publishes a correctly named event."""
        mock_action = MagicMock(spec=ActionInterface)
        mock_action.action_id = "test_action"
        action_plan = ActionPlanComponent(action_type=mock_action)
        event_data = {"action_plan_component": action_plan}

        action_system.on_action_chosen(event_data)

        mock_event_bus.publish.assert_called_once_with(
            "execute_test_action_action", event_data
        )

    def test_on_action_outcome_ready_full_cycle(
        self,
        action_system,
        mock_simulation_state,
        mock_event_bus,
        mock_action_cost_provider,
    ):
        """Test the full cycle from outcome to final event publishing."""
        entity_id = "agent_1"
        mock_action = MagicMock(spec=ActionInterface)
        mock_action.action_id = "test_action"
        mock_action.name = "Test Action"
        mock_action.get_base_cost.return_value = 1.0

        action_plan = ActionPlanComponent(
            action_type=mock_action, intent=MagicMock(name="TEST_INTENT")
        )
        action_outcome = ActionOutcome(
            success=True, message="Success", base_reward=10.0
        )

        mock_simulation_state.entities.get.return_value = {}

        event_data = {
            "entity_id": entity_id,
            "action_outcome": action_outcome,
            "original_action_plan": action_plan,
            "current_tick": 100,
        }

        action_system.on_action_outcome_ready(event_data)

        mock_action_cost_provider.apply_action_cost.assert_called_once_with(
            entity_id, 1.0, mock_simulation_state
        )

        # Assert that the publish method was called with the correct event name
        # and ANY object as the payload. This correctly tests the behavior.
        mock_event_bus.publish.assert_called_with("action_executed", ANY)

    @pytest.mark.asyncio
    async def test_update_method_is_empty(self, action_system):
        """Ensure the update method does nothing, as the system is event-driven."""
        await action_system.update(current_tick=1)

    def test_on_action_chosen_handles_invalid_plan(self, action_system, mock_event_bus):
        """Verify that no event is dispatched if the action plan is invalid."""
        action_plan = ActionPlanComponent(action_type=None)
        event_data = {"action_plan_component": action_plan}

        action_system.on_action_chosen(event_data)

        mock_event_bus.publish.assert_not_called()

    def test_on_action_outcome_ready_handles_invalid_plan(
        self, action_system, mock_event_bus
    ):
        """Verify the system doesn't crash with an invalid original plan."""
        action_outcome = ActionOutcome(
            success=True, message="Success", base_reward=10.0
        )
        action_plan = ActionPlanComponent(action_type=None)  # Invalid
        event_data = {
            "entity_id": "agent_1",
            "action_outcome": action_outcome,
            "original_action_plan": action_plan,
            "current_tick": 100,
        }

        action_system.on_action_outcome_ready(event_data)
        mock_event_bus.publish.assert_not_called()
