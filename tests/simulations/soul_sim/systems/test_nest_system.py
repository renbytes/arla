# tests/simulations/soul_sim/systems/test_nest_system.py
"""
Unit tests for the NestSystem in the soul_sim package.
"""

from unittest.mock import MagicMock, create_autospec

import pytest
from agent_engine.simulation.simulation_state import SimulationState
from omegaconf import OmegaConf

from simulations.soul_sim.components import InventoryComponent, NestComponent, PositionComponent
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.systems.nest_system import NestSystem


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

    system = NestSystem(
        simulation_state=mock_state,
        config=pydantic_config,
        cognitive_scaffold=MagicMock(),
    )
    system.event_bus = mock_bus
    return system, mock_state, mock_bus


def test_nest_creation_success(system_setup):
    """Tests that a nest is successfully created when an agent has enough resources."""
    system, mock_state, mock_bus = system_setup
    agent_id = "agent_1"

    mock_state.get_component.side_effect = [
        InventoryComponent(initial_resources=150.0),
        PositionComponent(position=(2, 2), environment=MagicMock()),
        NestComponent(),
    ]

    event_data = {"entity_id": agent_id, "action_plan_component": MagicMock(), "current_tick": 5}
    system.on_execute_nest_action(event_data)

    mock_bus.publish.assert_called_once()
    call_args = mock_bus.publish.call_args[0]
    outcome = call_args[1]["action_outcome"]
    assert call_args[1]["current_tick"] == 5
    assert outcome.success is True
    assert outcome.details["status"] == "nest_built"
    assert outcome.details["location"] == (2, 2)
    assert outcome.base_reward == 5.0  # Check hardcoded bonus


def test_nest_creation_fails_insufficient_resources(system_setup):
    """Tests that nest creation fails if the agent lacks the required resources."""
    system, mock_state, mock_bus = system_setup
    agent_id = "agent_1"

    mock_state.get_component.side_effect = [
        InventoryComponent(initial_resources=50.0),  # Not enough
        PositionComponent(position=(2, 2), environment=MagicMock()),
        NestComponent(),
    ]

    event_data = {"entity_id": agent_id, "action_plan_component": MagicMock(), "current_tick": 5}
    system.on_execute_nest_action(event_data)

    mock_bus.publish.assert_called_once()
    outcome = mock_bus.publish.call_args[0][1]["action_outcome"]
    assert outcome.success is False
    assert outcome.details["status"] == "failed_insufficient_resources"
    assert outcome.base_reward == -0.1


def test_nest_creation_fails_missing_component(system_setup):
    """Tests that the system exits gracefully if an agent is missing a required component."""
    system, mock_state, mock_bus = system_setup
    mock_state.get_component.return_value = None  # Simulate missing component

    event_data = {"entity_id": "agent_missing_comp", "action_plan_component": MagicMock(), "current_tick": 1}
    system.on_execute_nest_action(event_data)

    mock_bus.publish.assert_not_called()


@pytest.mark.asyncio
async def test_update_is_empty(system_setup):
    """Verifies that the async update method can be called without error as it's empty."""
    system, _, _ = system_setup
    try:
        await system.update(current_tick=100)
    except Exception as e:
        pytest.fail(f"NestSystem.update() raised an unexpected exception: {e}")
