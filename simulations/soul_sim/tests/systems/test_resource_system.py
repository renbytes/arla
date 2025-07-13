# simulations/soul_sim/tests/systems/test_resource_system.py
"""
Unit tests for the ResourceSystem in the soul_sim package.
"""

from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import ActionPlanComponent

# Components used by the system
from simulations.soul_sim.components import (
    InventoryComponent,
    PositionComponent,
    ResourceComponent,
)

# Subject under test
from simulations.soul_sim.systems.resource_system import ResourceSystem

# --- Test Fixtures ---


@pytest.fixture
def system_setup():
    """Pytest fixture to set up a test environment for the ResourceSystem."""
    mock_event_bus = MagicMock()
    mock_sim_state = MagicMock()
    mock_sim_state.event_bus = mock_event_bus

    # --- Define Agent and Resource Entities for Testing ---
    miner_comps = {
        InventoryComponent: InventoryComponent(initial_resources=10.0),
        PositionComponent: PositionComponent(position=(5, 5), environment=MagicMock()),
    }
    resource_healthy_comps = {
        ResourceComponent: ResourceComponent("test_res", 50, 1, 1, 10, 2.0, 20, 50),
        PositionComponent: PositionComponent(position=(5, 5), environment=MagicMock()),
    }
    resource_depletable_comps = {
        ResourceComponent: ResourceComponent("test_res", 10, 1, 1, 10, 2.0, 20, 50),
        PositionComponent: PositionComponent(position=(5, 5), environment=MagicMock()),
    }
    resource_depleted_comps = {
        ResourceComponent: ResourceComponent("test_res", 0, 1, 1, 10, 2.0, 20, 50),
        PositionComponent: PositionComponent(position=(5, 5), environment=MagicMock()),
    }
    resource_depleted_comps[ResourceComponent].is_depleted = True

    mock_sim_state.entities = {
        "miner": miner_comps,
        "resource_healthy": resource_healthy_comps,
        "resource_depletable": resource_depletable_comps,
        "resource_depleted": resource_depleted_comps,
    }

    # Configure the mock's get_component method
    def get_component_side_effect(entity_id, component_type):
        return mock_sim_state.entities.get(entity_id, {}).get(component_type)

    mock_sim_state.get_component.side_effect = get_component_side_effect

    system = ResourceSystem(simulation_state=mock_sim_state, config=MagicMock(), cognitive_scaffold=MagicMock())

    return system, mock_sim_state, mock_event_bus


# --- Test Cases for on_execute_extract ---


def test_extract_success_no_depletion(system_setup):
    """Tests a standard, successful mining action that does not deplete the resource."""
    system, mock_sim_state, mock_event_bus = system_setup
    miner_inv = mock_sim_state.entities["miner"][InventoryComponent]
    res_comp = mock_sim_state.entities["resource_healthy"][ResourceComponent]
    action_plan = ActionPlanComponent(params={"resource_id": "resource_healthy"})
    event_data = {"entity_id": "miner", "action_plan_component": action_plan, "current_tick": 1}

    system.on_execute_extract(event_data)

    assert res_comp.current_health == 40  # 50 - 10
    assert miner_inv.current_resources == 12.0  # 10 + 2.0
    mock_event_bus.publish.assert_called_once()
    outcome = mock_event_bus.publish.call_args.kwargs["event_data"]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "mining_progress"


def test_extract_success_with_depletion(system_setup):
    """Tests a mining action that successfully depletes the resource."""
    system, mock_sim_state, mock_event_bus = system_setup
    miner_inv = mock_sim_state.entities["miner"][InventoryComponent]
    res_comp = mock_sim_state.entities["resource_depletable"][ResourceComponent]
    action_plan = ActionPlanComponent(params={"resource_id": "resource_depletable"})
    event_data = {"entity_id": "miner", "action_plan_component": action_plan, "current_tick": 1}

    system.on_execute_extract(event_data)

    assert res_comp.current_health == 0
    assert res_comp.is_depleted is True
    # Initial 10 + 2.0 reward + 20 yield
    assert miner_inv.current_resources == 32.0
    mock_event_bus.publish.assert_called_once()
    outcome = mock_event_bus.publish.call_args.kwargs["event_data"]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "resource_depleted"


def test_extract_fails_if_already_depleted(system_setup):
    """Tests that mining a depleted resource fails."""
    system, _, mock_event_bus = system_setup
    action_plan = ActionPlanComponent(params={"resource_id": "resource_depleted"})
    event_data = {"entity_id": "miner", "action_plan_component": action_plan, "current_tick": 1}

    system.on_execute_extract(event_data)

    mock_event_bus.publish.assert_called_once()
    outcome = mock_event_bus.publish.call_args.kwargs["event_data"]["action_outcome"]
    assert outcome.success is False
    assert outcome.details["status"] == "invalid_mine_target"


def test_extract_fails_if_wrong_location(system_setup):
    """Tests that mining fails if the agent is not at the resource's location."""
    system, mock_sim_state, mock_event_bus = system_setup
    # Move the miner away from the resource
    mock_sim_state.entities["miner"][PositionComponent].position = (0, 0)
    action_plan = ActionPlanComponent(params={"resource_id": "resource_healthy"})
    event_data = {"entity_id": "miner", "action_plan_component": action_plan, "current_tick": 1}

    system.on_execute_extract(event_data)

    mock_event_bus.publish.assert_called_once()
    outcome = mock_event_bus.publish.call_args.kwargs["event_data"]["action_outcome"]
    assert outcome.success is False
    assert outcome.details["status"] == "invalid_mine_target"


# --- Test Cases for update (Respawning) ---


@pytest.mark.asyncio
async def test_update_does_nothing_on_non_interval_tick(system_setup):
    """Tests that the update method exits early if not on a 10th tick."""
    system, mock_sim_state, _ = system_setup
    await system.update(current_tick=9)
    mock_sim_state.get_entities_with_components.assert_not_called()


@pytest.mark.asyncio
async def test_update_respawns_resource(system_setup):
    """Tests that a depleted resource with a finished timer is respawned."""
    system, mock_sim_state, _ = system_setup
    res_comp = mock_sim_state.entities["resource_depleted"][ResourceComponent]
    res_comp.depleted_timer = 40  # Set timer so it will be >= respawn_time after update

    # Configure mock to return the depleted resource
    mock_sim_state.get_entities_with_components.return_value = {
        "resource_depleted": mock_sim_state.entities["resource_depleted"]
    }

    await system.update(current_tick=10)

    assert res_comp.is_depleted is False
    assert res_comp.current_health == res_comp.initial_health
    assert res_comp.depleted_timer == 0


@pytest.mark.asyncio
async def test_update_increments_timer(system_setup):
    """Tests that a depleted resource's timer is incremented if not ready to respawn."""
    system, mock_sim_state, _ = system_setup
    res_comp = mock_sim_state.entities["resource_depleted"][ResourceComponent]
    res_comp.depleted_timer = 20  # Set timer so it will not be ready

    mock_sim_state.get_entities_with_components.return_value = {
        "resource_depleted": mock_sim_state.entities["resource_depleted"]
    }

    await system.update(current_tick=20)

    assert res_comp.is_depleted is True
    assert res_comp.depleted_timer == 30  # 20 + 10
