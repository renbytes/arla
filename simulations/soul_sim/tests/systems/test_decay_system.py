# agent-engine/tests/systems/test_decay_system.py
"""
Unit tests for the DecaySystem in the agent-engine package.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState

from simulations.soul_sim.components import HealthComponent, InventoryComponent
from simulations.soul_sim.systems.decay_system import DecaySystem

# --- Test Fixtures ---


@pytest.fixture
def system_setup():
    """Pytest fixture to set up a test environment for the DecaySystem."""
    mock_event_bus = MagicMock()
    mock_sim_state = MagicMock(spec=SimulationState)
    mock_sim_state.event_bus = mock_event_bus

    # Create a mock config object with the required nested structure
    mock_config = SimpleNamespace(
        agent=SimpleNamespace(
            dynamics=SimpleNamespace(
                decay=SimpleNamespace(
                    time_budget_per_step=0.5,
                    health_per_step=1.0,
                    resources_per_step=0.1,
                )
            )
        )
    )

    # --- Define Agent Archetypes for Testing ---
    agent_active = {
        TimeBudgetComponent: TimeBudgetComponent(100.0, 0),
        HealthComponent: HealthComponent(100.0),
        InventoryComponent: InventoryComponent(10.0),
    }
    agent_inactive = {
        TimeBudgetComponent: TimeBudgetComponent(100.0, 0),
        HealthComponent: HealthComponent(100.0),
    }
    agent_inactive[TimeBudgetComponent].is_active = False

    agent_no_inventory = {
        TimeBudgetComponent: TimeBudgetComponent(50.0, 0),
        HealthComponent: HealthComponent(50.0),
    }
    agent_dies_by_time = {
        TimeBudgetComponent: TimeBudgetComponent(0.4, 0),
        HealthComponent: HealthComponent(100.0),
    }
    agent_dies_by_health = {
        TimeBudgetComponent: TimeBudgetComponent(100.0, 0),
        HealthComponent: HealthComponent(0.9),
    }

    mock_sim_state.entities = {
        "agent_active": agent_active,
        "agent_inactive": agent_inactive,
        "agent_no_inventory": agent_no_inventory,
        "agent_dies_by_time": agent_dies_by_time,
        "agent_dies_by_health": agent_dies_by_health,
    }

    # The system uses this method to get the entities to process
    mock_sim_state.get_entities_with_components.return_value = {
        k: v for k, v in mock_sim_state.entities.items() if v[TimeBudgetComponent].is_active
    }

    system = DecaySystem(
        simulation_state=mock_sim_state,
        config=mock_config,
        cognitive_scaffold=MagicMock(),
    )

    return system, mock_sim_state, mock_event_bus


# --- Test Cases ---


@pytest.mark.asyncio
async def test_decay_active_agent_with_inventory(system_setup):
    """
    Tests that an active agent with all components has its vitals correctly decayed.
    """
    system, _, _ = system_setup
    agent_comps = system.simulation_state.entities["agent_active"]
    time_comp = agent_comps[TimeBudgetComponent]
    health_comp = agent_comps[HealthComponent]
    inv_comp = agent_comps[InventoryComponent]

    await system.update(current_tick=1)

    assert time_comp.current_time_budget == pytest.approx(100.0 - 0.5)
    assert health_comp.current_health == pytest.approx(100.0 - 1.0)
    assert inv_comp.current_resources == pytest.approx(10.0 - 0.1)
    assert time_comp.is_active is True


@pytest.mark.asyncio
async def test_decay_active_agent_no_inventory(system_setup):
    """
    Tests that an agent without an InventoryComponent is decayed correctly without error.
    """
    system, _, _ = system_setup
    agent_comps = system.simulation_state.entities["agent_no_inventory"]
    time_comp = agent_comps[TimeBudgetComponent]
    health_comp = agent_comps[HealthComponent]

    await system.update(current_tick=1)

    assert time_comp.current_time_budget == pytest.approx(50.0 - 0.5)
    assert health_comp.current_health == pytest.approx(50.0 - 1.0)
    assert time_comp.is_active is True


@pytest.mark.asyncio
async def test_system_skips_inactive_agent(system_setup):
    """
    Tests that an agent with is_active=False is not affected by the decay system.
    """
    system, _, _ = system_setup
    agent_comps = system.simulation_state.entities["agent_inactive"]
    time_comp = agent_comps[TimeBudgetComponent]
    health_comp = agent_comps[HealthComponent]

    await system.update(current_tick=1)

    # Vitals should be unchanged
    assert time_comp.current_time_budget == 100.0
    assert health_comp.current_health == 100.0


@pytest.mark.asyncio
async def test_inactivation_by_time_depletion(system_setup):
    """
    Tests that an agent is correctly inactivated when its time budget reaches zero.
    """
    system, mock_sim_state, mock_event_bus = system_setup
    system.simulation_state.get_entities_with_components.return_value = {
        "agent_dies_by_time": mock_sim_state.entities["agent_dies_by_time"]
    }
    agent_comps = system.simulation_state.entities["agent_dies_by_time"]
    time_comp = agent_comps[TimeBudgetComponent]

    await system.update(current_tick=50)

    assert time_comp.is_active is False
    assert time_comp.current_time_budget == 0
    mock_event_bus.publish.assert_called_once_with(
        "entity_inactivated",
        {"entity_id": "agent_dies_by_time", "current_tick": 50, "reason": "time budget depletion"},
    )


@pytest.mark.asyncio
async def test_inactivation_by_health_depletion(system_setup):
    """
    Tests that an agent is correctly inactivated when its health reaches zero.
    """
    system, mock_sim_state, mock_event_bus = system_setup
    system.simulation_state.get_entities_with_components.return_value = {
        "agent_dies_by_health": mock_sim_state.entities["agent_dies_by_health"]
    }
    agent_comps = system.simulation_state.entities["agent_dies_by_health"]
    time_comp = agent_comps[TimeBudgetComponent]
    health_comp = agent_comps[HealthComponent]

    await system.update(current_tick=99)

    assert time_comp.is_active is False
    assert health_comp.current_health == 0
    mock_event_bus.publish.assert_called_once_with(
        "entity_inactivated",
        {"entity_id": "agent_dies_by_health", "current_tick": 99, "reason": "health depletion"},
    )


@pytest.mark.asyncio
async def test_inactivation_no_event_bus(system_setup):
    """
    Tests that the system does not crash if the event bus is not available.
    """
    system, mock_sim_state, mock_event_bus = system_setup
    system.simulation_state.get_entities_with_components.return_value = {
        "agent_dies_by_time": mock_sim_state.entities["agent_dies_by_time"]
    }
    # Remove the event bus from the system
    system.event_bus = None

    await system.update(current_tick=50)

    # Assert the agent is still inactivated
    time_comp = system.simulation_state.entities["agent_dies_by_time"][TimeBudgetComponent]
    assert time_comp.is_active is False
    # Assert that publish was never called
    mock_event_bus.publish.assert_not_called()
