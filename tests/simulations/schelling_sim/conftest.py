from typing import Any, Optional
from unittest.mock import Mock

import pytest

from simulations.schelling_sim.environment import SchellingGridEnvironment


class MockSchellingEnvironment(SchellingGridEnvironment):
    """A concrete mock implementation of SchellingGridEnvironment for testing."""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        # Initialize internal state for the mock
        self.grid = {}
        self.entity_positions = {}

    def can_move(self, from_pos: Any, to_pos: Any) -> bool:
        return True

    def remove_entity(self, entity_id: str) -> None:
        if entity_id in self.entity_positions:
            pos = self.entity_positions.pop(entity_id)
            if pos in self.grid:
                del self.grid[pos]

    def update_entity_position(
        self, entity_id: str, old_pos: Optional[Any], new_pos: Any
    ) -> None:
        # Simple update logic for the mock
        if old_pos in self.grid:
            del self.grid[old_pos]
        self.grid[new_pos] = entity_id
        self.entity_positions[entity_id] = new_pos


@pytest.fixture
def mock_simulation_state():
    """Provides a mock simulation state with a mock environment."""
    mock_state = Mock()
    mock_state.environment = Mock(spec=SchellingGridEnvironment)
    mock_state.config = {"satisfaction_threshold": 0.5}
    return mock_state


@pytest.fixture
def mock_schelling_environment(mock_simulation_state: Mock) -> Mock:
    """Fixture to provide a mock SchellingGridEnvironment."""
    mock_env = mock_simulation_state.environment
    mock_env.width = 10
    mock_env.height = 10
    mock_env.grid = {}
    mock_env.entity_positions = {}
    mock_env.get_valid_positions.return_value = [
        (x, y) for x in range(10) for y in range(10)
    ]
    mock_env.get_empty_cells.return_value = [
        (x, y) for x in range(10) for y in range(10)
    ]
    return mock_env


@pytest.fixture
def populated_environment(mock_schelling_environment: Mock) -> Mock:
    """Fixture to provide a mock environment with some agents placed."""
    mock_schelling_environment.grid = {
        (0, 0): "agent_A",
        (0, 1): "agent_B",
        (1, 0): "agent_C",
        (1, 1): "agent_D",
    }
    mock_schelling_environment.entity_positions = {
        "agent_A": (0, 0),
        "agent_B": (0, 1),
        "agent_C": (1, 0),
        "agent_D": (1, 1),
    }
    mock_schelling_environment.get_empty_cells.return_value = [
        (x, y)
        for x in range(10)
        for y in range(10)
        if (x, y) not in mock_schelling_environment.grid
    ]
    return mock_schelling_environment
