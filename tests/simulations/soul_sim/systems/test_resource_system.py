# tests/simulations/soul_sim/systems/test_resource_system.py
"""
Unit tests for the ResourceSystem in the soul_sim package.
"""

from unittest.mock import MagicMock, create_autospec

import pytest
from agent_engine.simulation.simulation_state import SimulationState

from simulations.soul_sim.components import (
    InventoryComponent,
    PositionComponent,
    ResourceComponent,
)
from simulations.soul_sim.systems.resource_system import ResourceSystem


@pytest.fixture
def system_setup():
    """Fixture to set up the system and its dependencies."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_bus = MagicMock()

    system = ResourceSystem(
        simulation_state=mock_state,
        config={},
        cognitive_scaffold=MagicMock(),
    )
    system.event_bus = mock_bus

    # CORRECTED: Create the depleted resource component correctly
    depleted_res_comp = ResourceComponent(
        resource_type="DOUBLE_NODE",
        initial_health=100.0,
        min_agents=2,
        max_agents=2,
        mining_rate=10.0,
        reward_per_mine=2.0,
        resource_yield=5.0,
        respawn_time=30,
    )
    depleted_res_comp.is_depleted = True  # Set attribute AFTER instantiation

    mock_state.entities = {
        "miner_1": {
            InventoryComponent: InventoryComponent(initial_resources=10.0),
            PositionComponent: PositionComponent(position=(1, 1), environment=MagicMock()),
        },
        "res_1": {
            ResourceComponent: ResourceComponent(
                resource_type="SINGLE_NODE",
                initial_health=50.0,
                min_agents=1,
                max_agents=1,
                mining_rate=10.0,
                reward_per_mine=2.0,
                resource_yield=5.0,
                respawn_time=50,
            ),
            PositionComponent: PositionComponent(position=(1, 1), environment=MagicMock()),
        },
        "res_depleted": {
            ResourceComponent: depleted_res_comp,
            PositionComponent: PositionComponent(position=(2, 2), environment=MagicMock()),
        },
    }
    return system, mock_state, mock_bus


def test_extract_success_no_depletion(system_setup):
    """Tests a successful mining action that doesn't deplete the resource."""
    system, mock_state, mock_bus = system_setup
    event_data = {
        "entity_id": "miner_1",
        "action_plan_component": MagicMock(params={"resource_id": "res_1"}),
        "current_tick": 10,
    }
    system.on_execute_extract(event_data)
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "mining_progress"


def test_extract_success_with_depletion(system_setup):
    """Tests a mining action that successfully depletes the resource."""
    system, mock_state, mock_bus = system_setup
    mock_state.entities["res_1"][ResourceComponent].current_health = 5.0

    event_data = {
        "entity_id": "miner_1",
        "action_plan_component": MagicMock(params={"resource_id": "res_1"}),
        "current_tick": 10,
    }
    system.on_execute_extract(event_data)
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "resource_depleted"


def test_extract_fails_if_already_depleted(system_setup):
    """Tests that mining a depleted resource fails."""
    system, mock_state, mock_bus = system_setup
    # Use the correctly-created depleted resource for this test
    event_data = {
        "entity_id": "miner_1",
        "action_plan_component": MagicMock(params={"resource_id": "res_depleted"}),
        "current_tick": 10,
    }
    system.on_execute_extract(event_data)
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is False
    assert "Resource is depleted" in outcome.message


@pytest.mark.asyncio
async def test_update_respawns_resource(system_setup):
    """Tests that a depleted resource with a finished timer is respawned."""
    system, mock_state, _ = system_setup
    res_comp = mock_state.entities["res_depleted"][ResourceComponent]
    res_comp.depleted_timer = 25
    mock_state.get_entities_with_components.return_value = {"res_depleted": mock_state.entities["res_depleted"]}

    await system.update(current_tick=20)  # A tick divisible by 10

    assert res_comp.is_depleted is False
    assert res_comp.current_health == res_comp.initial_health


@pytest.mark.asyncio
async def test_update_increments_timer(system_setup):
    """Tests that a depleted resource's timer is incremented if not ready to respawn."""
    system, mock_state, _ = system_setup
    res_comp = mock_state.entities["res_depleted"][ResourceComponent]
    res_comp.depleted_timer = 10
    mock_state.get_entities_with_components.return_value = {"res_depleted": mock_state.entities["res_depleted"]}

    await system.update(current_tick=10)

    assert res_comp.depleted_timer == 20
