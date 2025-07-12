# tests/unit/test_combat_system.py
"""
Unit tests for the CombatSystem in the soul_sim package.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from agent_core.core.ecs.component import ActionPlanComponent, TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState

from simulations.soul_sim.components import (
    CombatComponent,
    HealthComponent,
    PositionComponent,
)

# --- Corrected Absolute Imports ---
from simulations.soul_sim.environment.grid_world import GridWorld
from simulations.soul_sim.systems.combat_system import CombatSystem
from simulations.soul_sim.tests.systems.utils import MockEventBus

# --- Mock Objects and Fixtures for Testing ---


@pytest.fixture
def combat_system_setup():
    """Pytest fixture to set up a test environment for the CombatSystem."""
    mock_bus = MockEventBus()
    mock_env = GridWorld(width=10, height=10)

    mock_state = MagicMock(spec=SimulationState)
    mock_state.event_bus = mock_bus
    mock_state.environment = mock_env
    mock_state.config = {
        "learning": {"rewards": {"combat_reward_hit": 0.1, "combat_reward_defeat": 10.0}},
    }
    mock_state.main_rng = np.random.default_rng(123)
    mock_state.entities = {}

    # Create Attacker
    attacker_id = "attacker_1"
    mock_state.entities[attacker_id] = {
        PositionComponent: PositionComponent(position=(5, 5), environment=mock_env),
        HealthComponent: HealthComponent(initial_health=100.0),
        CombatComponent: CombatComponent(attack_power=10.0),
        TimeBudgetComponent: TimeBudgetComponent(initial_time_budget=100.0, lifespan_std_dev_percent=0.0),
    }

    # Create Defender
    defender_id = "defender_1"
    mock_state.entities[defender_id] = {
        PositionComponent: PositionComponent(position=(5, 6), environment=mock_env),
        HealthComponent: HealthComponent(initial_health=100.0),
        CombatComponent: CombatComponent(attack_power=5.0),
        TimeBudgetComponent: TimeBudgetComponent(initial_time_budget=100.0, lifespan_std_dev_percent=0.0),
    }

    def get_component_side_effect(entity_id, comp_type):
        return mock_state.entities.get(entity_id, {}).get(comp_type)

    mock_state.get_component.side_effect = get_component_side_effect

    system = CombatSystem(simulation_state=mock_state, config=mock_state.config, cognitive_scaffold=None)

    return system, mock_state, mock_bus, attacker_id, defender_id


# --- Test Cases for CombatSystem ---


def test_combat_system_resolves_attack(combat_system_setup):
    """
    Tests that the CombatSystem correctly resolves an attack, reduces defender's
    health, and publishes a success outcome.
    """
    system, mock_state, mock_bus, attacker_id, defender_id = combat_system_setup

    initial_defender_health = mock_state.get_component(defender_id, HealthComponent).current_health

    action_plan = ActionPlanComponent(params={"target_agent_id": defender_id})

    # FIX: The system expects the key "action_plan_component" from the event.
    combat_event = {
        "entity_id": attacker_id,
        "action_plan_component": action_plan,
        "current_tick": 1,
    }

    system.on_execute_combat(combat_event)

    # --- Assertions ---
    # 1. Check that the defender's health was reduced
    final_defender_health = mock_state.get_component(defender_id, HealthComponent).current_health
    assert final_defender_health < initial_defender_health

    # 2. Check that a successful outcome was published for the attacker
    published_event = mock_bus.get_last_published_event()
    assert published_event is not None
    assert published_event["type"] == "action_outcome_ready"

    outcome_data = published_event["data"]["action_outcome"]
    assert outcome_data.success is True
    assert outcome_data.details["status"] == "hit_target"
    assert outcome_data.details["damage_dealt"] > 0


def test_combat_system_handles_defeat(combat_system_setup):
    """
    Tests that the CombatSystem correctly handles a lethal attack, setting the
    defender's health to 0 and their status to inactive.
    """
    system, mock_state, mock_bus, attacker_id, defender_id = combat_system_setup

    # Give the attacker a very high attack power to ensure defeat
    attacker_combat_comp = mock_state.get_component(attacker_id, CombatComponent)
    attacker_combat_comp.attack_power = 200.0

    action_plan = ActionPlanComponent(params={"target_agent_id": defender_id})
    # FIX: The system expects the key "action_plan_component" from the event.
    combat_event = {
        "entity_id": attacker_id,
        "action_plan_component": action_plan,
        "current_tick": 1,
    }

    system.on_execute_combat(combat_event)

    # --- Assertions ---
    # 1. Defender's health should be 0
    final_defender_health = mock_state.get_component(defender_id, HealthComponent).current_health
    assert final_defender_health == 0

    # 2. Defender should be inactive
    defender_time_comp = mock_state.get_component(defender_id, TimeBudgetComponent)
    assert defender_time_comp.is_active is False

    # 3. Attacker's outcome should reflect the victory
    published_event = mock_bus.get_last_published_event()
    outcome_data = published_event["data"]["action_outcome"]
    assert outcome_data.details["status"] == "defeated_entity"
