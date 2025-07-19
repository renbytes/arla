# tests/simulations/soul_sim/systems/test_decay_system.py
"""
Unit tests for the DecaySystem.
"""

from unittest.mock import MagicMock, create_autospec

import pytest
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState
from omegaconf import OmegaConf

from simulations.soul_sim.components import HealthComponent
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.systems.decay_system import DecaySystem


@pytest.fixture
def pydantic_config():
    """Loads a STABLE, test-specific config."""
    conf = OmegaConf.load("tests/simulations/soul_sim/test_config.yml")
    return SoulSimAppConfig(**OmegaConf.to_container(conf, resolve=True))


@pytest.fixture
def system_setup(pydantic_config):
    """A comprehensive fixture for the DecaySystem."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_bus = MagicMock()

    system = DecaySystem(
        simulation_state=mock_state,
        config=pydantic_config,
        cognitive_scaffold=MagicMock(),
    )
    system.event_bus = mock_bus

    inactive_time_comp = TimeBudgetComponent(100.0, 0.0)
    inactive_time_comp.is_active = False

    mock_state.entities = {
        "agent_dies_health": {
            TimeBudgetComponent: TimeBudgetComponent(100.0, 0.0),
            HealthComponent: HealthComponent(5.0),  # Low health
        },
    }
    return system, mock_state, mock_bus


@pytest.mark.asyncio
async def test_inactivation_by_health_depletion(system_setup):
    """
    Tests that an agent is correctly inactivated when its health reaches zero,
    and the correct reason is published.
    """
    system, mock_state, mock_bus = system_setup
    # Set a high decay rate to ensure health depletion is the cause
    system.config.agent.dynamics.decay.health_per_step = 10.0
    mock_state.get_entities_with_components.return_value = {
        "agent_dies_health": mock_state.entities["agent_dies_health"]
    }

    # ACT
    await system.update(current_tick=99)

    # ASSERT
    agent_comps = mock_state.entities["agent_dies_health"]
    assert agent_comps[TimeBudgetComponent].is_active is False
    assert agent_comps[HealthComponent].current_health == 0

    # With the bug fixed in DecaySystem, this assertion will now pass
    mock_bus.publish.assert_called_once_with(
        "entity_inactivated",
        {
            "entity_id": "agent_dies_health",
            "current_tick": 99,
            "reason": "health depletion",
        },
    )
