# tests/unit/test_movement_system.py

from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import ActionPlanComponent
from agent_engine.simulation.simulation_state import SimulationState

from simulations.soul_sim.components import PositionComponent
from simulations.soul_sim.environment.grid_world import GridWorld
from simulations.soul_sim.systems.movement_system import MovementSystem
from simulations.soul_sim.tests.systems.utils import MockEventBus

# --- Mock Objects and Fixtures for Testing ---


@pytest.fixture
def movement_system_setup():
    """Pytest fixture to set up a test environment for the MovementSystem."""
    mock_bus = MockEventBus()
    mock_env = GridWorld(width=10, height=10)

    mock_state = MagicMock(spec=SimulationState)
    mock_state.event_bus = mock_bus
    mock_state.environment = mock_env
    mock_state.config = {
        "learning": {"rewards": {"exploration_base_reward": 0.5}},
    }
    mock_state.entities = {}

    # Create an entity
    entity_id = "agent_1"
    pos_component = PositionComponent(position=(5, 5), environment=mock_env)
    mock_state.entities[entity_id] = {PositionComponent: pos_component}

    def get_component_side_effect(entity_id, comp_type):
        return mock_state.entities.get(entity_id, {}).get(comp_type)

    mock_state.get_component.side_effect = get_component_side_effect

    # Instantiate the system with mock state
    system = MovementSystem(simulation_state=mock_state, config=mock_state.config, cognitive_scaffold=None)

    return system, mock_state, mock_bus, entity_id


# --- Test Cases for MovementSystem ---


def test_movement_system_moves_entity(movement_system_setup):
    """
    Tests that the MovementSystem correctly updates an entity's position
    when a valid move event is processed.
    """
    system, mock_state, mock_bus, entity_id = movement_system_setup

    action_plan = ActionPlanComponent(params={"direction": 0})  # Move Up
    # The system expects the key "action_plan_component" from the event.
    move_event = {
        "entity_id": entity_id,
        "action_plan_component": action_plan,
        "current_tick": 1,
    }

    system.on_execute_move(move_event)

    updated_pos_comp = mock_state.get_component(entity_id, PositionComponent)
    assert updated_pos_comp.position == (4, 5)

    published_event = mock_bus.get_last_published_event()
    assert published_event is not None
    assert published_event["type"] == "action_outcome_ready"

    outcome_data = published_event["data"]["action_outcome"]
    assert outcome_data.success is True
    assert outcome_data.details["status"] == "moved"


def test_movement_system_handles_boundary(movement_system_setup):
    """
    Tests that the MovementSystem prevents an entity from moving out of bounds
    and publishes a failure outcome.
    """
    system, mock_state, mock_bus, entity_id = movement_system_setup

    pos_component = mock_state.get_component(entity_id, PositionComponent)
    pos_component.position = (0, 5)

    action_plan = ActionPlanComponent(params={"direction": 0})  # Move Up
    # The system expects the key "action_plan_component" from the event.
    move_event = {
        "entity_id": entity_id,
        "action_plan_component": action_plan,
        "current_tick": 1,
    }

    system.on_execute_move(move_event)

    updated_pos_comp = mock_state.get_component(entity_id, PositionComponent)
    assert updated_pos_comp.position == (0, 5)

    published_event = mock_bus.get_last_published_event()
    assert published_event["type"] == "action_outcome_ready"

    outcome_data = published_event["data"]["action_outcome"]
    assert outcome_data.success is False
    assert outcome_data.details["status"] == "movement_blocked"
