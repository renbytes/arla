# simulations/soul_sim/tests/systems/test_nest_system.py
"""
Unit tests for the NestSystem in the soul_sim package.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import ActionPlanComponent

# Components used by the system
from simulations.soul_sim.components import (
    InventoryComponent,
    NestComponent,
    PositionComponent,
)

# Subject under test
from simulations.soul_sim.systems.nest_system import NestSystem

# --- Test Fixtures ---


@pytest.fixture
def system_setup():
    """Pytest fixture to set up a test environment for the NestSystem."""
    mock_event_bus = MagicMock()
    mock_sim_state = MagicMock()
    mock_sim_state.event_bus = mock_event_bus

    # Create a mock config object with the required nested structure
    mock_config = SimpleNamespace(learning=SimpleNamespace(rewards=SimpleNamespace(nest_resource_cost=100.0)))

    # --- Define Agent Archetypes for Testing ---
    agent_can_afford = {
        InventoryComponent: InventoryComponent(initial_resources=150.0),
        PositionComponent: PositionComponent(position=(5, 5), environment=MagicMock()),
        NestComponent: NestComponent(),
    }
    agent_cannot_afford = {
        InventoryComponent: InventoryComponent(initial_resources=50.0),
        PositionComponent: PositionComponent(position=(1, 1), environment=MagicMock()),
        NestComponent: NestComponent(),
    }
    agent_missing_comp = {
        InventoryComponent: InventoryComponent(initial_resources=200.0),
        PositionComponent: PositionComponent(position=(2, 2), environment=MagicMock()),
        # This agent is missing the NestComponent
    }

    mock_sim_state.entities = {
        "agent_can_afford": agent_can_afford,
        "agent_cannot_afford": agent_cannot_afford,
        "agent_missing_comp": agent_missing_comp,
    }

    # Configure the mock's get_component method to return the correct component for each agent
    def get_component_side_effect(entity_id, component_type):
        return mock_sim_state.entities.get(entity_id, {}).get(component_type)

    mock_sim_state.get_component.side_effect = get_component_side_effect

    system = NestSystem(
        simulation_state=mock_sim_state,
        config=mock_config,
        cognitive_scaffold=MagicMock(),
    )

    return system, mock_sim_state, mock_event_bus


# --- Test Cases ---


def test_nest_creation_success(system_setup):
    """
    Tests that an agent with sufficient resources can successfully build a nest.
    """
    system, mock_sim_state, mock_event_bus = system_setup
    agent_id = "agent_can_afford"
    agent_comps = mock_sim_state.entities[agent_id]
    inv_comp = agent_comps[InventoryComponent]
    nest_comp = agent_comps[NestComponent]
    pos_comp = agent_comps[PositionComponent]

    event_data = {
        "entity_id": agent_id,
        "action_plan_component": ActionPlanComponent(),
        "current_tick": 10,
    }

    system.on_execute_nest_action(event_data)

    # Assert resource cost was deducted
    assert inv_comp.current_resources == 150.0 - 100.0
    # Assert nest location was added
    assert len(nest_comp.locations) == 1
    assert nest_comp.locations[0] == pos_comp.position

    # Assert the correct outcome was published
    mock_event_bus.publish.assert_called_once()
    published_event_data = mock_event_bus.publish.call_args.kwargs["event_data"]
    outcome = published_event_data["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "nest_built"


def test_nest_creation_fails_insufficient_resources(system_setup):
    """
    Tests that an agent with insufficient resources fails to build a nest.
    """
    system, mock_sim_state, mock_event_bus = system_setup
    agent_id = "agent_cannot_afford"
    agent_comps = mock_sim_state.entities[agent_id]
    inv_comp = agent_comps[InventoryComponent]
    nest_comp = agent_comps[NestComponent]

    event_data = {
        "entity_id": agent_id,
        "action_plan_component": ActionPlanComponent(),
        "current_tick": 10,
    }

    system.on_execute_nest_action(event_data)

    # Assert resources and nests are unchanged
    assert inv_comp.current_resources == 50.0
    assert len(nest_comp.locations) == 0

    # Assert the correct failure outcome was published
    mock_event_bus.publish.assert_called_once()
    published_event_data = mock_event_bus.publish.call_args.kwargs["event_data"]
    outcome = published_event_data["action_outcome"]
    assert outcome.success is False
    assert outcome.details["status"] == "failed_insufficient_resources"


def test_nest_creation_fails_missing_component(system_setup):
    """
    Tests that the system exits gracefully if an agent is missing a required component.
    """
    system, _, mock_event_bus = system_setup
    event_data = {"entity_id": "agent_missing_comp", "action_plan_component": MagicMock()}

    system.on_execute_nest_action(event_data)

    # Assert that no outcome event was published because the system returned early
    mock_event_bus.publish.assert_not_called()


def test_publish_outcome_no_event_bus(system_setup):
    """
    Tests that the system does not crash if the event bus is None.
    """
    system, _, mock_event_bus = system_setup
    system.event_bus = None  # Manually remove the event bus

    event_data = {
        "entity_id": "agent_cannot_afford",
        "action_plan_component": ActionPlanComponent(),
        "current_tick": 10,
    }

    try:
        system.on_execute_nest_action(event_data)
    except Exception as e:
        pytest.fail(f"System crashed when event_bus was None: {e}")

    # Assert that no event was published
    mock_event_bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_update_is_empty(system_setup):
    """
    Verifies that the async update method can be called without error.
    """
    system, _, _ = system_setup
    try:
        await system.update(current_tick=100)
    except Exception as e:
        pytest.fail(f"NestSystem.update() raised an unexpected exception: {e}")
