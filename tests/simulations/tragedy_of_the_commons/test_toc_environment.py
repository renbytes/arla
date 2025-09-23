"""
Unit tests for the CommonsEnvironment in the Tragedy of the Commons simulation.

This file tests the environment's core functionalities, such as grid management,
agent placement, and resource queries, ensuring the simulation world behaves
as expected.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from simulations.tragedy_of_the_commons.environment import CommonsEnvironment
from simulations.tragedy_of_the_commons.components import ResourceComponent


@pytest.fixture
def mock_rng():
    """Provides a mock NumPy random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def commons_env(mock_rng: np.random.Generator) -> CommonsEnvironment:
    """Provides a standard CommonsEnvironment instance for tests."""
    return CommonsEnvironment(width=10, height=10, rng=mock_rng)


class TestCommonsEnvironment:
    """Tests for the CommonsEnvironment class."""

    def test_initialization(self, commons_env: CommonsEnvironment):
        """Test that the environment initializes with the correct dimensions."""
        assert commons_env.width == 10
        assert commons_env.height == 10
        assert isinstance(commons_env.rng, np.random.Generator)
        assert not commons_env.agent_positions
        assert not commons_env.resource_grid

    def test_update_and_remove_entity(self, commons_env: CommonsEnvironment):
        """Test adding, updating, and removing an agent's position."""
        # Add entity
        commons_env.update_entity_position("agent_1", None, (5, 5))
        assert commons_env.agent_positions["agent_1"] == (5, 5)
        assert commons_env.is_occupied_by_agent((5, 5))

        # Update entity position
        commons_env.update_entity_position("agent_1", (5, 5), (5, 6))
        assert commons_env.agent_positions["agent_1"] == (5, 6)
        assert not commons_env.is_occupied_by_agent((5, 5))
        assert commons_env.is_occupied_by_agent((5, 6))

        # Remove entity
        commons_env.remove_entity("agent_1")
        assert "agent_1" not in commons_env.agent_positions
        assert not commons_env.is_occupied_by_agent((5, 6))

    def test_get_resource_at_with_mock_state(self, commons_env: CommonsEnvironment):
        """Test querying resource levels using a mocked simulation state."""
        # Setup mock simulation state and component
        mock_sim_state = MagicMock()
        resource_comp = ResourceComponent(
            current_resource=15.0, max_resource=20, regeneration_rate=0.1
        )
        mock_sim_state.get_component.return_value = resource_comp

        # Link the mock state to the environment
        commons_env.simulation_state = mock_sim_state
        commons_env.resource_grid[(3, 3)] = "grass_patch_1"

        # Test getting resource
        assert commons_env.get_resource_at((3, 3)) == 15.0
        mock_sim_state.get_component.assert_called_with(
            "grass_patch_1", ResourceComponent
        )

        # Test getting resource from an empty location
        assert commons_env.get_resource_at((4, 4)) == 0.0

    def test_consume_resource_with_mock_state(self, commons_env: CommonsEnvironment):
        """Test consuming resources using a mocked simulation state."""
        mock_sim_state = MagicMock()
        resource_comp = ResourceComponent(
            current_resource=15.0, max_resource=20, regeneration_rate=0.1
        )
        mock_sim_state.get_component.return_value = resource_comp

        commons_env.simulation_state = mock_sim_state
        commons_env.resource_grid[(3, 3)] = "grass_patch_1"

        consumed = commons_env.consume_resource((3, 3), 10.0)
        assert consumed == 10.0
        assert resource_comp.current_resource == 5.0

    def test_get_all_empty_cells(self, commons_env: CommonsEnvironment):
        """Test finding all empty cells on the grid."""
        commons_env.update_entity_position("agent_1", None, (0, 0))
        commons_env.update_entity_position("agent_2", None, (0, 1))
        empty_cells = commons_env.get_all_empty_cells()
        assert len(empty_cells) == 98
        assert (0, 0) not in empty_cells
        assert (0, 1) not in empty_cells
        assert (1, 1) in empty_cells
