# tests/simulations/schelling_sim/test_systems.py

import asyncio
from unittest.mock import Mock

import pytest

from simulations.schelling_sim.components import (
    GroupComponent,
    PositionComponent,
    SatisfactionComponent,
)
from simulations.schelling_sim.systems import SatisfactionSystem


@pytest.fixture
def mock_simulation_state():
    """Provides a mock SimulationState for testing."""
    from simulations.schelling_sim.environment import SchellingGridEnvironment

    state = Mock()
    # Create a real SchellingGridEnvironment instance but mock its methods
    state.environment = SchellingGridEnvironment(width=10, height=10)
    state.environment.get_neighbors_of_position = Mock()
    state.get_component = Mock()
    state.get_entities_with_components = Mock()
    return state


@pytest.fixture
def satisfaction_system(mock_simulation_state):
    """Provides an instance of the SatisfactionSystem with a mock state."""
    # This system does not depend on config or the scaffold
    return SatisfactionSystem(mock_simulation_state, None, None)


def test_update_satisfaction_becomes_satisfied(satisfaction_system, mock_simulation_state):
    """Tests an agent becomes satisfied when its neighbor ratio meets the threshold."""
    # Arrange
    agent_id = "agent_A"
    pos_comp = PositionComponent(x=1, y=1)
    group_comp = GroupComponent(agent_type=1)
    satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.5)
    satisfaction_comp.is_satisfied = False

    # Store all components in a structure that mimics the real SimulationState
    all_comps = {
        "agent_A": {
            PositionComponent: pos_comp,
            GroupComponent: group_comp,
            SatisfactionComponent: satisfaction_comp,
        },
        "neighbor1": {GroupComponent: GroupComponent(agent_type=1)},
        "neighbor2": {GroupComponent: GroupComponent(agent_type=1)},
        "neighbor3": {GroupComponent: GroupComponent(agent_type=2)},
    }

    # Configure mocks to use the component store
    mock_simulation_state.get_entities_with_components.return_value = {agent_id: all_comps[agent_id]}
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {
        (1, 0): "neighbor1",
        (1, 2): "neighbor2",
        (0, 1): "neighbor3",
    }
    mock_simulation_state.get_component.side_effect = lambda eid, ctype: all_comps.get(eid, {}).get(ctype)

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert (2 out of 3 neighbors are the same type = 66.7% > 50% threshold)
    assert satisfaction_comp.is_satisfied is True


def test_update_satisfaction_becomes_unsatisfied(satisfaction_system, mock_simulation_state):
    """Tests an agent becomes unsatisfied when its neighbor ratio falls below the threshold."""
    # Arrange
    agent_id = "agent_A"
    pos_comp = PositionComponent(x=1, y=1)
    group_comp = GroupComponent(agent_type=1)
    satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.5)
    satisfaction_comp.is_satisfied = True

    all_comps = {
        "agent_A": {
            PositionComponent: pos_comp,
            GroupComponent: group_comp,
            SatisfactionComponent: satisfaction_comp,
        },
        "neighbor1": {GroupComponent: GroupComponent(agent_type=1)},
        "neighbor2": {GroupComponent: GroupComponent(agent_type=2)},
        "neighbor3": {GroupComponent: GroupComponent(agent_type=2)},
    }

    mock_simulation_state.get_entities_with_components.return_value = {agent_id: all_comps[agent_id]}
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {
        (1, 0): "neighbor1",
        (1, 2): "neighbor2",
        (0, 1): "neighbor3",
    }
    mock_simulation_state.get_component.side_effect = lambda eid, ctype: all_comps.get(eid, {}).get(ctype)

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert (1 out of 3 neighbors are the same type = 33.3% < 50% threshold)
    assert satisfaction_comp.is_satisfied is False


def test_update_isolated_agent_is_satisfied(satisfaction_system, mock_simulation_state):
    """Tests that an agent with no neighbors is always satisfied."""
    # Arrange
    agent_id = "agent_A"
    pos_comp = PositionComponent(x=1, y=1)
    group_comp = GroupComponent(agent_type=1)
    satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.5)
    satisfaction_comp.is_satisfied = False

    all_comps = {
        "agent_A": {
            PositionComponent: pos_comp,
            GroupComponent: group_comp,
            SatisfactionComponent: satisfaction_comp,
        }
    }

    mock_simulation_state.get_entities_with_components.return_value = {agent_id: all_comps[agent_id]}
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {}

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert
    assert satisfaction_comp.is_satisfied is True


def test_update_agent_with_high_threshold_becomes_unsatisfied(satisfaction_system, mock_simulation_state):
    """Tests that an agent's specific threshold is respected."""
    # Arrange
    agent_id = "agent_A"
    pos_comp = PositionComponent(x=1, y=1)
    group_comp = GroupComponent(agent_type=1)
    satisfaction_comp = SatisfactionComponent(satisfaction_threshold=0.7)
    satisfaction_comp.is_satisfied = True

    all_comps = {
        "agent_A": {
            PositionComponent: pos_comp,
            GroupComponent: group_comp,
            SatisfactionComponent: satisfaction_comp,
        },
        "neighbor1": {GroupComponent: GroupComponent(agent_type=1)},
        "neighbor2": {GroupComponent: GroupComponent(agent_type=1)},
        "neighbor3": {GroupComponent: GroupComponent(agent_type=2)},
    }

    mock_simulation_state.get_entities_with_components.return_value = {agent_id: all_comps[agent_id]}
    mock_simulation_state.environment.get_neighbors_of_position.return_value = {
        (1, 0): "neighbor1",
        (1, 2): "neighbor2",
        (0, 1): "neighbor3",
    }
    mock_simulation_state.get_component.side_effect = lambda eid, ctype: all_comps.get(eid, {}).get(ctype)

    # Act
    asyncio.run(satisfaction_system.update(current_tick=1))

    # Assert (2 out of 3 neighbors are same type = 66.7% < 70% threshold)
    assert satisfaction_comp.is_satisfied is False
