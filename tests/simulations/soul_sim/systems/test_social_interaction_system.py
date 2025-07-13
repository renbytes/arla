# tests/simulations/soul_sim/systems/test_social_interaction_system.py
"""
Comprehensive unit tests for the SocialInteractionSystem in the soul_sim package.
"""

from unittest.mock import MagicMock, create_autospec

import pytest
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import ActionPlanComponent, TimeBudgetComponent
from agent_engine.simulation.simulation_state import SimulationState
from omegaconf import OmegaConf

from simulations.soul_sim.components import EnvironmentObservationComponent, PositionComponent
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.systems.social_interaction_system import SocialInteractionSystem


@pytest.fixture
def pydantic_config():
    """Loads the base config into a validated Pydantic model."""
    conf = OmegaConf.load("simulations/soul_sim/config/base_config.yml")
    return SoulSimAppConfig(**OmegaConf.to_container(conf, resolve=True))


@pytest.fixture
def system_setup(pydantic_config):
    """Fixture to set up the system and its dependencies for tests."""
    mock_state = create_autospec(SimulationState, instance=True)
    mock_bus = MagicMock()
    mock_state.environment = MagicMock()
    mock_state.environment.distance.return_value = 1

    system = SocialInteractionSystem(
        simulation_state=mock_state,
        config=pydantic_config,
        cognitive_scaffold=MagicMock(),
    )
    system.event_bus = mock_bus

    # Correctly create the inactive component
    inactive_time_comp = TimeBudgetComponent(100.0, 0.0)
    inactive_time_comp.is_active = False

    mock_state.entities = {
        "agent_a": {
            TimeBudgetComponent: TimeBudgetComponent(100.0, 0.0),
            PositionComponent: PositionComponent((0, 0), mock_state.environment),
            EnvironmentObservationComponent: EnvironmentObservationComponent(),
        },
        "agent_b": {
            TimeBudgetComponent: TimeBudgetComponent(100.0, 0.0),
            PositionComponent: PositionComponent((0, 1), mock_state.environment),
            EnvironmentObservationComponent: EnvironmentObservationComponent(),
        },
        "agent_c": {  # Inactive agent
            TimeBudgetComponent: inactive_time_comp,
            PositionComponent: PositionComponent((1, 1), mock_state.environment),
        },
    }
    return system, mock_state, mock_bus


def test_successful_cooperative_communication(system_setup):
    """Tests a cooperative communication action succeeds and has the correct reward."""
    system, _, mock_bus = system_setup
    event_data = {
        "entity_id": "agent_a",
        "action_plan_component": ActionPlanComponent(params={"target_agent_id": "agent_b"}, intent=Intent.COOPERATE),
        "current_tick": 1,
    }
    system.on_execute_communicate(event_data)
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is True
    assert outcome.base_reward == 2.5


def test_successful_competitive_communication(system_setup):
    """Tests a competitive communication action succeeds with a different reward."""
    system, _, mock_bus = system_setup
    event_data = {
        "entity_id": "agent_a",
        "action_plan_component": ActionPlanComponent(params={"target_agent_id": "agent_b"}, intent=Intent.COMPETE),
        "current_tick": 1,
    }
    system.on_execute_communicate(event_data)
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is True
    assert outcome.base_reward == 0.5


@pytest.mark.parametrize(
    "target_id, setup_func, expected_reason",
    [
        (
            "agent_b",
            lambda state: setattr(state.environment, "distance", MagicMock(return_value=5)),
            "Target is too far away.",
        ),
        ("agent_c", lambda state: None, "Target is inactive."),
        ("non_existent_agent", lambda state: None, "Communicator or target does not exist."),
    ],
)
def test_failed_communication_due_to_validation(system_setup, target_id, setup_func, expected_reason):
    """Tests various validation failure scenarios for communication."""
    system, mock_state, mock_bus = system_setup
    setup_func(mock_state)
    event_data = {
        "entity_id": "agent_a",
        "action_plan_component": ActionPlanComponent(params={"target_agent_id": target_id}, intent=Intent.COOPERATE),
        "current_tick": 1,
    }
    system.on_execute_communicate(event_data)
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is False
    assert outcome.details["reason"] == expected_reason
