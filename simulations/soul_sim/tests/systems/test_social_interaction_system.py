# src/simulations/soul_sim/tests/test_social_interaction_system.py
"""
Comprehensive unit tests for the SocialInteractionSystem in the soul_sim package.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import ActionPlanComponent, TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState
from simulations.soul_sim.components import (
    EnvironmentObservationComponent,
    PositionComponent,
)
from simulations.soul_sim.environment.grid_world import GridWorld
from simulations.soul_sim.systems.social_interaction_system import (
    SocialInteractionSystem,
)
from simulations.soul_sim.tests.systems.utils import MockEventBus
from simulations.soul_sim.world import Grid2DEnvironment

# --- Mock Objects and Fixtures ---

@pytest.fixture
def system_setup():
    """Pytest fixture to set up a test environment for the SocialInteractionSystem."""
    mock_bus = MockEventBus()
    mock_env = GridWorld(width=20, height=20)
    mock_state = MagicMock(spec=SimulationState)
    mock_state.event_bus = mock_bus
    mock_state.environment = mock_env
    mock_state.config = {
        "learning": {
            "rewards": {
                "communicate_reward_base": 0.05,
                "collaboration_bonus_per_agent": 0.1,
            }
        }
    }
    mock_state.entities = {}

    # --- Create Agents ---
    # Agent A (The Communicator)
    mock_state.entities["agent_a"] = {
        PositionComponent: PositionComponent((5, 5), mock_env),
        TimeBudgetComponent: TimeBudgetComponent(100.0, 0.0),
        EnvironmentObservationComponent: EnvironmentObservationComponent()
    }
    mock_state.entities["agent_a"][EnvironmentObservationComponent].known_entity_locations["enemy_1"] = (1, 1)


    # Agent B (The Target)
    mock_state.entities["agent_b"] = {
        PositionComponent: PositionComponent((5, 6), mock_env),
        TimeBudgetComponent: TimeBudgetComponent(100.0, 0.0),
        EnvironmentObservationComponent: EnvironmentObservationComponent()
    }
    mock_state.entities["agent_b"][EnvironmentObservationComponent].known_entity_locations["resource_1"] = (10, 10)

    # Agent C (Inactive Target)
    time_c = TimeBudgetComponent(100.0, 0.0)
    time_c.is_active = False
    mock_state.entities["agent_c"] = {
        PositionComponent: PositionComponent((5, 7), mock_env),
        TimeBudgetComponent: time_c
    }

    def get_component_side_effect(entity_id, comp_type):
        return mock_state.entities.get(entity_id, {}).get(comp_type)
    mock_state.get_component.side_effect = get_component_side_effect

    system = SocialInteractionSystem(
        simulation_state=mock_state,
        config=mock_state.config,
        cognitive_scaffold=MagicMock(),
    )

    return system, mock_state, mock_bus


# --- Test Cases ---


def test_successful_cooperative_communication(system_setup):
    """
    Tests that a COOPERATE intent successfully triggers information exchange
    and publishes a success outcome with a bonus reward.
    """
    system, mock_state, mock_bus = system_setup
    action_plan = ActionPlanComponent(params={"target_agent_id": "agent_b"}, intent=Intent.COOPERATE)
    # FIX: The system expects the key "action_plan_component" from the event.
    event_data = {"entity_id": "agent_a", "action_plan_component": action_plan, "current_tick": 1}

    # ACT
    system.on_execute_communicate(event_data)

    # ASSERT
    # 1. Information was exchanged
    obs_a = mock_state.get_component("agent_a", EnvironmentObservationComponent)
    obs_b = mock_state.get_component("agent_b", EnvironmentObservationComponent)
    assert "resource_1" in obs_a.known_entity_locations
    assert "enemy_1" in obs_b.known_entity_locations

    # 2. A successful outcome was published
    published_event = mock_bus.get_last_published_event()
    assert published_event["type"] == "action_outcome_ready"
    outcome = published_event["data"]["action_outcome"]
    assert outcome.success is True
    assert outcome.details["status"] == "communicated"

    # 3. Reward includes the collaboration bonus
    expected_reward = 0.05 + 0.1
    assert outcome.base_reward == pytest.approx(expected_reward)


def test_successful_competitive_communication(system_setup):
    """
    Tests that a non-cooperative intent succeeds but does NOT trigger
    information exchange or add a bonus reward.
    """
    system, mock_state, mock_bus = system_setup
    action_plan = ActionPlanComponent(params={"target_agent_id": "agent_b"}, intent=Intent.COMPETE)
    # FIX: The system expects the key "action_plan_component" from the event.
    event_data = {"entity_id": "agent_a", "action_plan_component": action_plan, "current_tick": 1}

    # ACT
    system.on_execute_communicate(event_data)

    # ASSERT
    # 1. Information was NOT exchanged
    obs_a = mock_state.get_component("agent_a", EnvironmentObservationComponent)
    obs_b = mock_state.get_component("agent_b", EnvironmentObservationComponent)
    assert "resource_1" not in obs_a.known_entity_locations
    assert "enemy_1" not in obs_b.known_entity_locations

    # 2. A successful outcome was published
    published_event = mock_bus.get_last_published_event()
    assert published_event["type"] == "action_outcome_ready"
    outcome = published_event["data"]["action_outcome"]
    assert outcome.success is True

    # 3. Reward is only the base reward
    assert outcome.base_reward == pytest.approx(0.05)


@pytest.mark.parametrize(
    "target_id, setup_func, expected_reason",
    [
        (
            "agent_b",
            lambda state: state.get_component("agent_b", PositionComponent).__setattr__("position", (15, 15)),
            "Target is too far away.",
        ),
        ("agent_c", lambda state: None, "Target is inactive."),
        ("non_existent_agent", lambda state: None, "Communicator or target does not exist."),
        (
            "agent_b",
            lambda state: state.entities["agent_b"].pop(PositionComponent),
            "Communicator or target is missing a PositionComponent.",
        ),
    ],
)
def test_failed_communication_due_to_validation(system_setup, target_id, setup_func, expected_reason):
    """
    Tests various validation failure scenarios for communication.
    """
    system, mock_state, mock_bus = system_setup

    # Apply the specific setup for this test case
    setup_func(mock_state)

    action_plan = ActionPlanComponent(params={"target_agent_id": target_id}, intent=Intent.COOPERATE)
    # FIX: The system expects the key "action_plan_component" from the event.
    event_data = {"entity_id": "agent_a", "action_plan_component": action_plan, "current_tick": 1}

    # ACT
    system.on_execute_communicate(event_data)

    # ASSERT
    published_event = mock_bus.get_last_published_event()
    assert published_event["type"] == "action_outcome_ready"
    outcome = published_event["data"]["action_outcome"]

    assert outcome.success is False
    assert outcome.details["status"] == "communication_failed"
    assert outcome.details["reason"] == expected_reason


def test_communication_with_no_target(system_setup):
    """
    Tests that the system handles an action plan with no target ID gracefully.
    """
    system, _, mock_bus = system_setup
    action_plan = ActionPlanComponent(params={}, intent=Intent.COOPERATE)
    # FIX: The system expects the key "action_plan_component" from the event.
    event_data = {"entity_id": "agent_a", "action_plan_component": action_plan, "current_tick": 1}

    # ACT
    system.on_execute_communicate(event_data)

    # ASSERT
    # No event should be published because the system returns early
    assert mock_bus.get_last_published_event() is None


def test_update_method_is_empty(system_setup):
    """
    Confirms that the system's per-tick update method does nothing, as it is purely event-driven.
    """
    system, _, _ = system_setup
    # This test primarily ensures the async method can be called without error
    import asyncio

    try:
        asyncio.run(system.update(current_tick=10))
    except Exception as e:
        pytest.fail(f"system.update() raised an unexpected exception: {e}")
