# tests/test_runners.py

import pytest
from unittest.mock import AsyncMock

# Subject under test
from agent_concurrent.runners import SerialSystemRunner, AsyncSystemRunner

# --- Mocks and Fixtures ---


class MockSystem:
    """A mock system that conforms to the SystemProtocol for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail
        # Use AsyncMock for the update method to track async calls
        self.update = AsyncMock()

        if self.should_fail:
            self.update.side_effect = RuntimeError(f"System '{self.name}' failed as designed.")

    def __repr__(self) -> str:
        return f"MockSystem(name='{self.name}')"


@pytest.fixture
def systems():
    """Provides a list of mock systems for testing."""
    return [MockSystem("A"), MockSystem("B")]


@pytest.fixture
def systems_with_failure():
    """Provides a list of mock systems where one is designed to fail."""
    return [MockSystem("A"), MockSystem("B", should_fail=True), MockSystem("C")]


# --- Test Cases for SerialSystemRunner ---


@pytest.mark.asyncio
async def test_serial_runner_executes_all_systems(systems):
    """
    Tests that the SerialSystemRunner calls the update method on all systems.
    """
    # Arrange
    runner = SerialSystemRunner()

    # Act
    await runner.run(systems, current_tick=10)

    # Assert
    for system in systems:
        system.update.assert_awaited_once_with(current_tick=10)


@pytest.mark.asyncio
async def test_serial_runner_handles_failure_and_continues(systems_with_failure, capsys):
    """
    Tests that the SerialSystemRunner continues to run subsequent systems
    even if one of them fails.
    """
    # Arrange
    runner = SerialSystemRunner()

    # Act
    await runner.run(systems_with_failure, current_tick=20)

    # Assert
    # The first system should have been called
    systems_with_failure[0].update.assert_awaited_once_with(current_tick=20)
    # The failing system should have been called
    systems_with_failure[1].update.assert_awaited_once_with(current_tick=20)
    # The system after the failure should also have been called
    systems_with_failure[2].update.assert_awaited_once_with(current_tick=20)

    # Check that the error was logged to the console
    captured = capsys.readouterr()
    # Combine stdout and stderr to check the full output.
    full_output = captured.out + captured.err
    assert "ERROR: System 'MockSystem(name='B')' failed" in full_output
    assert "RuntimeError: System 'B' failed as designed." in full_output


# --- Test Cases for AsyncSystemRunner ---


@pytest.mark.asyncio
async def test_async_runner_executes_all_systems(systems):
    """
    Tests that the AsyncSystemRunner calls the update method on all systems concurrently.
    """
    # Arrange
    runner = AsyncSystemRunner()

    # Act
    await runner.run(systems, current_tick=30)

    # Assert
    for system in systems:
        system.update.assert_awaited_once_with(current_tick=30)


@pytest.mark.asyncio
async def test_async_runner_handles_failure_gracefully(systems_with_failure, capsys):
    """
    Tests that the AsyncSystemRunner waits for all tasks to complete and
    logs any exceptions, even when one system fails.
    """
    # Arrange
    runner = AsyncSystemRunner()

    # Act
    await runner.run(systems_with_failure, current_tick=40)

    # Assert
    # All systems, including the one after the failure, should have been awaited.
    systems_with_failure[0].update.assert_awaited_once_with(current_tick=40)
    systems_with_failure[1].update.assert_awaited_once_with(current_tick=40)
    systems_with_failure[2].update.assert_awaited_once_with(current_tick=40)

    # Check that the error was logged to the console after all tasks completed
    captured = capsys.readouterr()
    # Combine stdout and stderr to check the full output.
    full_output = captured.out + captured.err
    assert "ERROR: System 'MockSystem(name='B')' failed" in full_output
    assert "RuntimeError: System 'B' failed as designed." in full_output
