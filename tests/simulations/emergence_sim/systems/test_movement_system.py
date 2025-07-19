from unittest.mock import MagicMock

import pytest
from agent_core.agents.actions.base_action import ActionOutcome

from simulations.emergence_sim.components import InventoryComponent, PositionComponent
from simulations.emergence_sim.systems.move_system import MovementSystem

# Fixtures


@pytest.fixture
def mock_simulation_state():
    """Provides a mocked SimulationState with an event bus and entity store."""
    state = MagicMock()
    state.event_bus = MagicMock()
    state.environment = MagicMock()
    state.entities = {}

    def get_component(entity_id, comp_type):
        return state.entities.get(entity_id, {}).get(comp_type)

    state.get_component = get_component

    return state


@pytest.fixture
def movement_system(mock_simulation_state):
    """Returns an instance of the MovementSystem."""
    return MovementSystem(
        simulation_state=mock_simulation_state,
        config=MagicMock(),
        cognitive_scaffold=MagicMock(),
    )


# MovementSystem Tests


def test_on_execute_move_updates_position_and_publishes_outcome(movement_system, mock_simulation_state):
    """
    Tests that a move event correctly updates the agent's PositionComponent
    and publishes a successful outcome.
    """
    # 1. ARRANGE
    agent_id = "agent_1"
    initial_pos = (5, 5)
    target_pos = (5, 6)

    # Set up the agent with a PositionComponent
    pos_comp = PositionComponent(position=initial_pos, environment=mock_simulation_state.environment)
    inv_comp = InventoryComponent(initial_resources=10.0)
    mock_simulation_state.entities[agent_id] = {
        PositionComponent: pos_comp,
        InventoryComponent: inv_comp,
    }

    # Create the event data that the system will receive
    event_data = {
        "entity_id": agent_id,
        "current_tick": 10,
        "action_plan_component": MagicMock(params={"new_pos": target_pos}),
    }

    # 2. ACT
    movement_system.on_execute_move(event_data)

    # 3. ASSERT
    # The agent's position component should be updated
    assert pos_comp.position == target_pos

    # The environment should be told to update its spatial index
    mock_simulation_state.environment.update_entity_position.assert_called_once_with(agent_id, initial_pos, target_pos)

    # An outcome event should be published
    mock_simulation_state.event_bus.publish.assert_called_once()
    call_args = mock_simulation_state.event_bus.publish.call_args[0]
    assert call_args[0] == "action_outcome_ready"
    published_outcome = call_args[1]["action_outcome"]
    assert isinstance(published_outcome, ActionOutcome)
    assert published_outcome.success is True
