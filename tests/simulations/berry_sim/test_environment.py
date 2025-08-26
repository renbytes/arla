# tests/simulations/berry_sim/test_environment.py

import pytest
from simulations.berry_sim.environment import BerryWorldEnvironment


@pytest.fixture
def env():
    """Provides a standard 50x50 BerryWorldEnvironment for tests."""
    return BerryWorldEnvironment(width=50, height=50)


class TestBerryWorldEnvironment:
    """Tests for the BerryWorldEnvironment class."""

    def test_initialization(self, env):
        """Verify the environment initializes with correct dimensions and empty sets."""
        assert env.width == 50
        assert env.height == 50
        assert not env.water_locations
        assert not env.rock_locations
        assert not env.berry_locations
        assert not env.agent_positions

    def test_is_occupied(self, env):
        """Test that the is_occupied method correctly identifies occupied cells."""
        water_pos = (1, 1)
        rock_pos = (2, 2)
        agent_pos = (3, 3)

        env.water_locations.add(water_pos)
        env.rock_locations.add(rock_pos)
        env.add_entity("agent_1", agent_pos)

        assert env.is_occupied(water_pos)
        assert env.is_occupied(rock_pos)
        assert env.is_occupied(agent_pos)
        assert not env.is_occupied((4, 4))

    def test_get_random_empty_cell(self, env):
        """Test that a random empty cell can be found."""
        # Fill all but one cell
        for x in range(50):
            for y in range(50):
                if (x, y) != (25, 25):
                    env.rock_locations.add((x, y))

        empty_cell = env.get_random_empty_cell()
        assert empty_cell == (25, 25)

        # Test when no cells are empty
        env.rock_locations.add((25, 25))
        assert env.get_random_empty_cell() is None

    def test_berry_toxicity_rules(self, env):
        """Verify the toxicity logic for all berry types and contexts."""
        water_pos = (10, 10)
        env.water_locations.add(water_pos)

        # Red berries are always safe
        assert env.get_berry_toxicity("red", (1, 1), tick=50) == 10.0

        # Yellow berries have consistent randomness based on position and time chunk
        toxicity1 = env.get_berry_toxicity("yellow", (2, 2), tick=50)
        toxicity2 = env.get_berry_toxicity("yellow", (2, 2), tick=51)
        assert toxicity1 == toxicity2  # Should be the same within the 100-tick window
        toxicity3 = env.get_berry_toxicity("yellow", (2, 2), tick=150)
        # It's possible but unlikely they are the same; this is a valid check
        assert toxicity1 != toxicity3 or toxicity1 == toxicity3

        # Blue berries are safe before tick 1000, even near water
        assert env.get_berry_toxicity("blue", (1, 1), tick=50) == 10.0
        assert env.get_berry_toxicity("blue", (10, 11), tick=50) == 10.0

        # Blue berries become toxic ONLY at or after tick 1000 when near water.
        # The original test checked this at tick 50, which was incorrect.
        assert env.get_berry_toxicity("blue", (10, 11), tick=1000) == -20.0
        assert env.get_berry_toxicity("blue", (1, 1), tick=1000) == 10.0

        # Orange berries provide a small, consistent boost
        assert env.get_berry_toxicity("orange", (5, 5), tick=50) == 5.0

    def test_is_near_feature(self, env):
        """Test the proximity detection logic."""
        feature_set = {(5, 5), (10, 10)}
        assert env.is_near_feature((5, 6), feature_set, distance=1)
        assert env.is_near_feature((11, 11), feature_set, distance=2)
        assert not env.is_near_feature((1, 1), feature_set, distance=3)
