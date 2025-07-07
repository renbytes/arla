# tests/simulation/test_system.py

from unittest.mock import MagicMock
import pytest

# Subject under test
from agent_engine.simulation.system import System, SystemManager


# --- Mocks and Fixtures ---


class MockSystemA(System):
    """A mock system for testing order and calls."""

    async def update(self, current_tick: int) -> None:
        # This attribute is added dynamically for tracking calls
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
    """
    Tests that a system can be registered and is stored internally.
    """
    # Act
    system_manager.register_system(MockSystemA)

    # Assert
    assert len(system_manager._systems) == 1
    assert isinstance(system_manager._systems[0], MockSystemA)


async def test_update_all_calls_systems_in_order(system_manager):
    """
    Tests that the update_all method calls the update method on each
    registered system in the order they were registered.
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

    assert hasattr(system_a, "update_called_at_tick")
    assert system_a.update_called_at_tick == 50

    assert hasattr(system_b, "update_called_at_tick")
    assert system_b.update_called_at_tick == 50


@pytest.mark.asyncio
async def test_update_all_continues_after_a_system_fails(system_manager, capsys):
    """
    Tests that if one system fails during its update, the manager logs
    the error and continues to update subsequent systems.
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
    assert system_b.update_called_at_tick == 99

    # Check that an error message was printed to the console
    captured = capsys.readouterr()
    full_output = captured.out + captured.err

    # Adjust the assertion string to match the runner's output.
    # The runner uses the system's __repr__ which is 'FailingSystem System'.
    expected_error_msg = "ERROR: System 'FailingSystem System' failed during concurrent update"
    assert expected_error_msg in full_output
    assert "RuntimeError: This system is designed to fail." in full_output


def test_get_system_retrieves_correct_instance(system_manager):
    """
    Tests that get_system can find and return a registered system instance by its class type.
    """
    # Arrange
    system_manager.register_system(MockSystemA)
    system_manager.register_system(MockSystemB)

    # Act
    retrieved_system = system_manager.get_system(MockSystemB)

    # Assert
    assert retrieved_system is not None
    assert isinstance(retrieved_system, MockSystemB)


def test_get_system_returns_none_for_unregistered_system(system_manager):
    """
    Tests that get_system returns None when asked for a system type that
    has not been registered.
    """
    # Arrange
    system_manager.register_system(MockSystemA)

    # Act
    retrieved_system = system_manager.get_system(MockSystemB)

    # Assert
    assert retrieved_system is None
