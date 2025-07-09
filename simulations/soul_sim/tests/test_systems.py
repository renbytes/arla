# tests/unit/test_systems.py
"""
Unit tests for the world-specific systems in the soul_sim package.
This file focuses on non-combat systems like movement.
"""

from typing import Any, Dict, List

import pytest
from agent_core.core.ecs.component import ActionPlanComponent
from agent_engine.simulation.simulation_state import SimulationState
from simulations.soul_sim.components import PositionComponent
from simulations.soul_sim.systems.movement_system import MovementSystem
from simulations.soul_sim.world import Grid2DEnvironment

# --- Mock Objects and Fixtures for Testing ---


class MockEventBus:
    """A mock event bus to capture published events for testing."""

    def __init__(self):
        self.published_events: List[Dict[str, Any]] = []

    def subscribe(self, event_type: str, handler):
        pass

    def publish(self, event_type: str, event_data: Dict[str, Any]):
        self.published_events.append({"type": event_type, "data": event_data})

    def get_last_published_event(self):
        return self.published_events[-1] if self.published_events else None


@pytest.fixture
def movement_system_setup():
    """Pytest fixture to set up a test environment for the MovementSystem."""
    mock_bus = MockEventBus()
    mock_env = Grid2DEnvironment(width=10, height=10)

    mock_state = SimulationState(config={}, device="cpu")
    mock_state.event_bus = mock_bus
    mock_state.environment = mock_env
    mock_state.config = {
        "learning": {"rewards": {"exploration_base_reward": 0.5}},
    }

    # Create an entity
    entity_id = "agent_1"
    pos_component = PositionComponent(position=(5, 5), environment=mock_env)
    mock_state.entities[entity_id] = {PositionComponent: pos_component}

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
    move_event = {"entity_id": entity_id, "action_plan": action_plan, "current_tick": 1}

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
    move_event = {"entity_id": entity_id, "action_plan": action_plan, "current_tick": 1}

    system.on_execute_move(move_event)

    updated_pos_comp = mock_state.get_component(entity_id, PositionComponent)
    assert updated_pos_comp.position == (0, 5)

    published_event = mock_bus.get_last_published_event()
    assert published_event["type"] == "action_outcome_ready"

    outcome_data = published_event["data"]["action_outcome"]
    assert outcome_data.success is False
    assert outcome_data.details["status"] == "movement_blocked"
