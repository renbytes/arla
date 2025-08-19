# tests/simulations/schelling_sim/test_environment.py

from unittest.mock import Mock

import pytest

from simulations.schelling_sim.components import PositionComponent
from simulations.schelling_sim.environment import SchellingGridEnvironment


@pytest.fixture
def env():
    """Provides a fresh 10x10 environment for each test."""
    return SchellingGridEnvironment(width=10, height=10)


def test_environment_initialization(env):
    assert env.width == 10
    assert env.height == 10
    assert env.grid == {}
    assert env.agent_positions == {}


def test_initialize_from_state(env):
    mock_sim_state = Mock()
    agent_components = {
        "agent_A": {PositionComponent: PositionComponent(x=1, y=1)},
        "agent_B": {PositionComponent: PositionComponent(x=2, y=2)},
    }
    env.initialize_from_state(mock_sim_state, agent_components)
    assert env.grid.get((1, 1)) == "agent_A"
    assert env.grid.get((2, 2)) == "agent_B"
    assert env.agent_positions["agent_A"] == (1, 1)


def test_get_neighbors_of_position_center(env):
    env.add_entity("agent_A", (5, 5))
    env.add_entity("agent_B", (5, 6))
    env.add_entity("agent_C", (4, 5))
    env.add_entity("agent_D", (6, 4))

    neighbors = env.get_neighbors_of_position((5, 5))

    # FIX: The test was asserting 2, but the setup clearly creates 3 neighbors.
    assert len(neighbors) == 3
    assert neighbors[(5, 6)] == "agent_B"
    assert neighbors[(4, 5)] == "agent_C"
    assert neighbors[(6, 4)] == "agent_D"


def test_get_neighbors_of_position_toroidal_wrap(env):
    env.add_entity("agent_A", (0, 0))
    env.add_entity("agent_B", (9, 0))
    env.add_entity("agent_C", (0, 9))
    env.add_entity("agent_D", (9, 9))
    env.add_entity("agent_E", (5, 5))

    neighbors = env.get_neighbors_of_position((0, 0))
    assert len(neighbors) == 3
    assert neighbors[(9, 0)] == "agent_B"
    assert neighbors[(0, 9)] == "agent_C"
    assert neighbors[(9, 9)] == "agent_D"


def test_get_empty_cells(env):
    env.add_entity("agent_A", (0, 0))
    env.add_entity("agent_B", (1, 1))
    empty_cells = env.get_empty_cells()
    assert len(empty_cells) == 98
    assert (0, 0) not in empty_cells
    assert (1, 1) not in empty_cells
    assert (5, 5) in empty_cells


def test_move_agent_success(env):
    env.add_entity("agent_A", (1, 1))
    success = env.move_entity("agent_A", from_pos=(1, 1), to_pos=(2, 2))
    assert success is True
    assert env.grid.get((1, 1)) is None
    assert env.grid.get((2, 2)) == "agent_A"
    assert env.agent_positions["agent_A"] == (2, 2)


def test_move_agent_to_occupied_cell(env):
    env.add_entity("agent_A", (1, 1))
    env.add_entity("agent_B", (2, 2))
    success = env.move_entity("agent_A", from_pos=(1, 1), to_pos=(2, 2))
    assert success is False
    assert env.grid.get((1, 1)) == "agent_A"
    assert env.grid.get((2, 2)) == "agent_B"


def test_distance_calculation_with_wrapping(env):
    pos1 = (1, 1)
    pos2 = (8, 9)
    dist = env.distance(pos1, pos2)
    assert dist == 5.0


def test_serialization_and_restoration(env):
    env.add_entity("agent_A", (1, 2))
    env.add_entity("agent_B", (3, 4))

    serialized_data = env.to_dict()

    new_env = SchellingGridEnvironment(width=1, height=1)
    new_env.restore_from_dict(serialized_data)

    assert new_env.width == env.width
    assert new_env.height == env.height
    assert new_env.grid == env.grid
    assert new_env.agent_positions == env.agent_positions
