# tests/simulations/soul_sim/systems/test_combat_system.py
"""
Unit tests for the CombatSystem in the soul_sim package.
"""

from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState
from omegaconf import OmegaConf

from simulations.soul_sim.components import (
    CombatComponent,
    HealthComponent,
    PositionComponent,
)
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.systems.combat_system import CombatSystem


@pytest.fixture
def pydantic_config():
    """Loads the base config into a validated Pydantic model for type-safe access."""
    conf = OmegaConf.load("tests/simulations/soul_sim/test_config.yml")
    return SoulSimAppConfig(**OmegaConf.to_container(conf, resolve=True))


@pytest.fixture
def combat_system(pydantic_config):
    """Initializes the CombatSystem with mocks and a validated config."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_bus = MagicMock()
    mock_state.entities = {}
    mock_state.main_rng = np.random.default_rng(42)
    mock_state.environment = MagicMock()
    mock_state.environment.distance.return_value = 1.0

    system = CombatSystem(
        simulation_state=mock_state,
        config=pydantic_config,
        cognitive_scaffold=MagicMock(),
    )
    system.event_bus = mock_bus
    return system, mock_state, mock_bus


def setup_combatants(sim_state, attacker_id, defender_id):
    """Helper to set up attacker and defender components."""
    sim_state.entities[attacker_id] = {
        CombatComponent: CombatComponent(attack_power=10.0),
        PositionComponent: PositionComponent(position=(0, 0), environment=sim_state.environment),
    }
    sim_state.entities[defender_id] = {
        HealthComponent: HealthComponent(initial_health=100.0),
        TimeBudgetComponent: TimeBudgetComponent(initial_time_budget=100.0, lifespan_std_dev_percent=0),
        PositionComponent: PositionComponent(position=(0, 1), environment=sim_state.environment),
    }


def test_combat_system_resolves_attack(combat_system):
    """Tests that a standard combat action correctly calculates damage and publishes an outcome."""
    system, mock_state, mock_bus = combat_system
    attacker_id, defender_id = "attacker_1", "defender_1"
    setup_combatants(mock_state, attacker_id, defender_id)

    event_data = {
        "entity_id": attacker_id,
        "action_plan_component": MagicMock(params={"target_agent_id": defender_id}),
        "current_tick": 1,
    }

    system.on_execute_combat(event_data)

    defender_health = mock_state.entities[defender_id][HealthComponent]
    assert defender_health.current_health < 100.0

    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args[0]
    assert call_args[0] == "action_outcome_ready"
    outcome = call_args[1]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "hit_target"
    assert outcome.base_reward == 5.0


def test_combat_system_handles_defeat(combat_system):
    """Tests that defeating an agent correctly sets their state to inactive and publishes the right events."""
    system, mock_state, mock_bus = combat_system
    attacker_id, defender_id = "attacker_1", "defender_1"
    setup_combatants(mock_state, attacker_id, defender_id)
    mock_state.entities[defender_id][HealthComponent].current_health = 5.0

    event_data = {
        "entity_id": attacker_id,
        "action_plan_component": MagicMock(params={"target_agent_id": defender_id}),
        "current_tick": 1,
    }

    system.on_execute_combat(event_data)

    defender_time_comp = mock_state.entities[defender_id][TimeBudgetComponent]
    assert defender_time_comp.is_active is False
    assert mock_state.entities[defender_id][HealthComponent].current_health == 0

    mock_bus.publish.assert_any_call(
        "entity_inactivated",
        {"entity_id": defender_id, "current_tick": 1},
    )

    call_args = mock_bus.publish.call_args[0]
    assert call_args[0] == "action_outcome_ready"
    outcome = call_args[1]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "defeated_entity"
    assert outcome.base_reward == 25.0
