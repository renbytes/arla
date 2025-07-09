# agent-engine/tests/simulation/test_system.py

from unittest.mock import MagicMock, AsyncMock
import pytest

# Subject under test
from agent_engine.simulation.system import System, SystemManager


# --- Mocks and Fixtures ---


class MockSystemA(System):
    """A mock system for testing order and calls."""

    async def update(self, current_tick: int) -> None:
        self.update_called_at_tick = current_tick


class MockSystemB(System):
    """Another mock system for testing order."""

    async def update(self, current_tick: int) -> None:
        self.update_called_at_tick = current_tick


class FailingSystem(System):
    """A mock system that is designed to raise an error."""

    async def update(self, current_tick: int) -> None:
        raise RuntimeError("This system is designed to fail.")


@pytest.fixture
def mock_simulation_state():
    """Mocks the SimulationState."""
    return MagicMock()


@pytest.fixture
def mock_cognitive_scaffold():
    """Mocks the CognitiveScaffold."""
    return MagicMock()


@pytest.fixture
def system_manager(mock_simulation_state, mock_cognitive_scaffold):
    """Provides a fresh SystemManager instance for each test."""
    return SystemManager(
        simulation_state=mock_simulation_state,
        config={},
        cognitive_scaffold=mock_cognitive_scaffold,
    )


# --- Test Cases ---


def test_register_system(system_manager):
    """Tests that a system can be registered and is stored internally."""
    system_manager.register_system(MockSystemA)
    assert len(system_manager._systems) == 1
    assert isinstance(system_manager._systems[0], MockSystemA)


@pytest.mark.asyncio
async def test_update_all_calls_systems_in_order(system_manager):
    """
    Tests that update_all calls the update method on each registered system.
    """
    # Arrange
    system_manager.register_system(MockSystemA)
    system_manager.register_system(MockSystemB)

    # Act
    await system_manager.update_all(current_tick=50)

    # Assert
    system_a = system_manager.get_system(MockSystemA)
    system_b = system_manager.get_system(MockSystemB)

    assert system_a is not None
    assert system_b is not None

    # Check that the attribute was set by the update method
    assert hasattr(system_a, "update_called_at_tick")
    assert system_a.update_called_at_tick == 50

    assert hasattr(system_b, "update_called_at_tick")
    assert system_b.update_called_at_tick == 50


@pytest.mark.asyncio
async def test_update_all_continues_after_a_system_fails(system_manager, capsys):
    """
    Tests that if one system fails, the manager continues to update others.
    """
    # Arrange
    system_manager.register_system(MockSystemA)
    system_manager.register_system(FailingSystem)
    system_manager.register_system(MockSystemB)

    # Act
    await system_manager.update_all(current_tick=99)

    # Assert
    # Check that the system after the failing one was still updated
    system_b = system_manager.get_system(MockSystemB)
    assert system_b is not None
    assert hasattr(system_b, "update_called_at_tick")
    assert system_b.update_called_at_tick == 99

    # Check that the runner logged the error
    captured = capsys.readouterr()
    full_output = captured.out + captured.err
    assert "ERROR: System 'FailingSystem System' failed" in full_output
    assert "RuntimeError: This system is designed to fail." in full_output


def test_get_system_retrieves_correct_instance(system_manager):
    """
    Tests that get_system can find and return a registered system instance.
    """
    system_manager.register_system(MockSystemA)
    system_manager.register_system(MockSystemB)
    retrieved_system = system_manager.get_system(MockSystemB)
    assert isinstance(retrieved_system, MockSystemB)


def test_get_system_returns_none_for_unregistered_system(system_manager):
    """
    Tests that get_system returns None for an unregistered system type.
    """
    system_manager.register_system(MockSystemA)
    retrieved_system = system_manager.get_system(MockSystemB)
    assert retrieved_system is None
