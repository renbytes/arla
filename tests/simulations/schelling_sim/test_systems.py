# tests/simulations/schelling_sim/test_systems.py

import asyncio
from unittest.mock import Mock

import pytest

from simulations.schelling_sim.components import (
    PositionComponent,
    SchellingAgentComponent,
)
from simulations.schelling_sim.systems import SatisfactionSystem


@pytest.fixture
def mock_simulation_state():
    """Provides a mock SimulationState for testing."""
    state = Mock()
    state.environment = Mock()
    # Add the required method to the mock environment
    state.environment.get_neighbors_of_position = Mock()
    state.get_component = Mock()
    state.get_entities_with_components = Mock()
    return state


@pytest.fixture
def satisfaction_system(mock_simulation_state):
    """Provides an instance of the SatisfactionSystem with a mock state."""
    mock_config = {"agent": {"satisfaction_threshold": 0.5}}
    mock_scaffold = Mock()
    return SatisfactionSystem(mock_simulation_state, mock_config, mock_scaffold)


def test_update_satisfaction_becomes_satisfied(satisfaction_system, mock_simulation_state):
    # Arrange
    agent_id = "agent_A"
    agent_comp = SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5)
    agent_comp.is_satisfied = False
    pos_comp = PositionComponent(x=1, y=1)

    # A single dictionary now holds all components for the mock.
    all_comps = {
        "agent_A": agent_comp,
        "neighbor1": SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5),
        "neighbor2": SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5),
        "neighbor3": SchellingAgentComponent(agent_type=2, satisfaction_threshold=0.5),
    }

    mock_simulation_state.get_entities_with_components.return_value = {
        agent_id: {
            PositionComponent: pos_comp,
            SchellingAgentComponent: agent_comp,
        }
    }
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {
        (1, 0): "neighbor1",
        (1, 2): "neighbor2",
        (0, 1): "neighbor3",
    }
    mock_simulation_state.get_component.side_effect = lambda eid, ctype: all_comps.get(eid)

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert
    assert agent_comp.is_satisfied is True


def test_update_satisfaction_becomes_unsatisfied(satisfaction_system, mock_simulation_state):
    # Arrange
    agent_id = "agent_A"
    agent_comp = SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5)
    agent_comp.is_satisfied = True
    pos_comp = PositionComponent(x=1, y=1)

    all_comps = {
        "agent_A": agent_comp,
        "neighbor1": SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5),
        "neighbor2": SchellingAgentComponent(agent_type=2, satisfaction_threshold=0.5),
        "neighbor3": SchellingAgentComponent(agent_type=2, satisfaction_threshold=0.5),
    }

    mock_simulation_state.get_entities_with_components.return_value = {
        agent_id: {
            PositionComponent: pos_comp,
            SchellingAgentComponent: agent_comp,
        }
    }
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {
        (1, 0): "neighbor1",
        (1, 2): "neighbor2",
        (0, 1): "neighbor3",
    }
    mock_simulation_state.get_component.side_effect = lambda eid, ctype: all_comps.get(eid)

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert
    assert agent_comp.is_satisfied is False


def test_update_satisfaction_isolated_agent_is_satisfied(satisfaction_system, mock_simulation_state):
    # Arrange
    agent_id = "agent_A"
    agent_comp = SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5)
    agent_comp.is_satisfied = False
    pos_comp = PositionComponent(x=1, y=1)

    mock_simulation_state.get_entities_with_components.return_value = {
        agent_id: {
            PositionComponent: pos_comp,
            SchellingAgentComponent: agent_comp,
        }
    }
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {}

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert
    assert agent_comp.is_satisfied is True


def test_update_uses_config_threshold(satisfaction_system, mock_simulation_state):
    # Arrange
    agent_id = "agent_A"
    agent_comp = SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.7)
    agent_comp.is_satisfied = True
    pos_comp = PositionComponent(x=1, y=1)

    all_comps = {
        "agent_A": agent_comp,
        "neighbor1": SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5),
        "neighbor2": SchellingAgentComponent(agent_type=1, satisfaction_threshold=0.5),
        "neighbor3": SchellingAgentComponent(agent_type=2, satisfaction_threshold=0.5),
    }

    mock_simulation_state.get_entities_with_components.return_value = {
        agent_id: {
            PositionComponent: pos_comp,
            SchellingAgentComponent: agent_comp,
        }
    }
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {
        (1, 0): "neighbor1",
        (1, 2): "neighbor2",
        (0, 1): "neighbor3",
    }
    mock_simulation_state.get_component.side_effect = lambda eid, ctype: all_comps.get(eid)

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert
    assert agent_comp.is_satisfied is False
