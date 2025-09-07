"""
Unit tests for the SugarscapeEnvironment.

These tests ensure that the environment's core mechanics—such as resource
distribution, regeneration, and agent placement—are functioning correctly
and are fully reproducible when given a specific random seed.
"""

import numpy as np
import pytest
from simulations.sugarscape_sim.environment import SugarscapeEnvironment


@pytest.fixture
def seeded_rng():
    """Provides a seeded random number generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def env(seeded_rng):
    """Provides a standard 50x50 SugarscapeEnvironment for tests."""
    return SugarscapeEnvironment(width=50, height=50, rng=seeded_rng)


class TestSugarscapeEnvironment:
    """Tests for the SugarscapeEnvironment class."""

    def test_initialization(self, env):
        """Verify the environment initializes with correct properties."""
        assert env.width == 50
        assert env.height == 50
        assert env.sugar_map.shape == (50, 50)
        assert not env.agent_positions

    def test_deterministic_sugar_distribution(self):
        """Verify that the same seed produces the identical sugar map."""
        rng1 = np.random.default_rng(123)
        env1 = SugarscapeEnvironment(width=10, height=10, rng=rng1)

        rng2 = np.random.default_rng(123)
        env2 = SugarscapeEnvironment(width=10, height=10, rng=rng2)

        np.testing.assert_array_equal(env1.sugar_map, env2.sugar_map)

    def test_sugar_regeneration(self, env):
        """Test that sugar regenerates correctly without exceeding the max."""
        env.sugar_map.fill(0)
        env.regenerate_sugar()
        assert np.all(env.sugar_map == env.sugar_regeneration_rate)

        env.sugar_map.fill(env.max_sugar_per_cell)
        env.regenerate_sugar()
        assert np.all(env.sugar_map == env.max_sugar_per_cell)

    def test_sugar_consumption(self, env):
        """Test that consuming sugar removes it from the map."""
        pos = (10, 10)
        initial_sugar = env.get_sugar_at(pos)

        consumed_sugar = env.consume_sugar(pos)

        assert consumed_sugar == initial_sugar
        assert env.get_sugar_at(pos) == 0

    def test_get_all_empty_cells(self, env):
        """Test that it correctly identifies all empty cells."""
        env.agent_positions["agent_1"] = (0, 0)
        env.agent_positions["agent_2"] = (1, 1)

        empty_cells = env.get_all_empty_cells()

        total_cells = env.width * env.height
        #  Changed from self.assertEqual to pytest-style assert
        assert len(empty_cells) == total_cells - 2
        assert (0, 0) not in empty_cells
        assert (1, 1) not in empty_cells

    def test_get_random_empty_cell_is_deterministic(self):
        """Verify that random cell selection is deterministic with the same seed."""
        rng1 = np.random.default_rng(42)
        env1 = SugarscapeEnvironment(width=10, height=10, rng=rng1)
        env1.agent_positions["agent_1"] = (5, 5)

        rng2 = np.random.default_rng(42)
        env2 = SugarscapeEnvironment(width=10, height=10, rng=rng2)
        env2.agent_positions["agent_1"] = (5, 5)

        cell1 = env1.get_random_empty_cell()
        cell2 = env2.get_random_empty_cell()

        assert cell1 == cell2

    def test_get_neighbors(self, env):
        """Test neighbor finding at various locations."""
        neighbors_center = env.get_neighbors((25, 25))
        assert len(neighbors_center) == 8

        neighbors_corner = env.get_neighbors((0, 0))
        assert len(neighbors_corner) == 3
        assert (1, 0) in neighbors_corner
        assert (0, 1) in neighbors_corner
        assert (1, 1) in neighbors_corner

        neighbors_edge = env.get_neighbors((0, 25))
        assert len(neighbors_edge) == 5
