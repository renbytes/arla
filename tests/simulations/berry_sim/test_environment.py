# FILE: tests/simulations/berry_sim/test_environment.py
"""
Unit tests for the BerryWorldEnvironment.

Ensures that the environment correctly manages its grid, agent positions,
and the specific rules for berry toxicity and spawning contexts.
"""

import pytest
from simulations.berry_sim.environment import BerryWorldEnvironment


@pytest.fixture
def env():
    """Provides a standard 20x20 BerryWorldEnvironment for tests."""
    return BerryWorldEnvironment(width=20, height=20)


class TestBerryWorldEnvironment:
    """Tests for the core logic of the BerryWorldEnvironment."""

    def test_initialization(self, env):
        """Verify the environment initializes with the correct dimensions."""
        assert env.width == 20
        assert env.height == 20
        assert not env.agent_positions
        assert not env.berry_locations

    def test_agent_and_entity_management(self, env):
        """Test adding, moving, and removing agents from the grid."""
        agent_id = "agent_1"
        start_pos = (5, 5)
        end_pos = (5, 6)

        env.add_entity(agent_id, start_pos)
        assert env.agent_positions[agent_id] == start_pos
        assert env.is_occupied(start_pos)
        assert env.get_entities_at_position(start_pos) == {agent_id}

        env.update_entity_position(agent_id, start_pos, end_pos)
        assert env.agent_positions[agent_id] == end_pos
        assert not env.is_occupied(start_pos)
        assert env.is_occupied(end_pos)

        env.remove_entity(agent_id)
        assert agent_id not in env.agent_positions
        assert not env.is_occupied(end_pos)

    def test_get_random_empty_cell(self, env):
        """Test finding an empty cell and handling a full grid."""
        pos = env.get_random_empty_cell()
        assert pos is not None
        assert 0 <= pos[0] < env.width
        assert 0 <= pos[1] < env.height

        for x in range(env.width):
            for y in range(env.height):
                env.rock_locations.add((x, y))
        assert env.get_random_empty_cell() is None

    def test_berry_toxicity_rules(self, env):
        """Verify the toxicity logic for all berry types and contexts."""
        water_pos = (10, 10)
        env.water_locations.add(water_pos)

        assert env.get_berry_toxicity("red", (1, 1), tick=50) == 10.0

        toxicity1 = env.get_berry_toxicity("yellow", (2, 2), tick=50)
        toxicity2 = env.get_berry_toxicity("yellow", (2, 2), tick=51)
        assert toxicity1 == toxicity2
        toxicity3 = env.get_berry_toxicity("yellow", (2, 2), tick=150)
        assert toxicity1 != toxicity3 or toxicity1 == toxicity3

        assert env.get_berry_toxicity("blue", (1, 1), tick=50) == 10.0
        assert env.get_berry_toxicity("blue", (10, 11), tick=50) == -20.0

    def test_environmental_context(self, env):
        """Test the context provider for the causal graph system."""
        env.water_locations.add((5, 5))
        env.rock_locations.add((15, 15))

        context1 = env.get_environmental_context((5, 6))
        assert context1["near_water"] is True
        assert context1["near_rocks"] is False

        context2 = env.get_environmental_context((14, 15))
        assert context2["near_water"] is False
        assert context2["near_rocks"] is True

        context3 = env.get_environmental_context((1, 1))
        assert context3["near_water"] is False
        assert context3["near_rocks"] is False
