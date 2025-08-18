# agent-engine/tests/simulation/test_system_manager.py

import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System, SystemManager


class MockSystem(System):
    """A mock system with an async update method for testing."""

    def __init__(self, simulation_state, config, cognitive_scaffold):
        super().__init__(simulation_state, config, cognitive_scaffold)
        # Replace the async update method with an AsyncMock
        self.update = AsyncMock()

    async def update(self, current_tick: int) -> None:
        """This method is mocked in __init__."""
        pass


@pytest.mark.asyncio
class TestSystemManager(unittest.TestCase):
    """
    Tests for the SystemManager to ensure correct registration and execution of systems.
    """

    def setUp(self):
        """Set up a mock simulation state and a SystemManager for each test."""
        self.simulation_state = MagicMock(spec=SimulationState)
        self.config = {}
        self.cognitive_scaffold = MagicMock()
        self.system_manager = SystemManager(self.simulation_state, self.config, self.cognitive_scaffold)

    def test_register_system(self):
        """Verify that a system can be registered with the manager."""
        mock_system_instance = MockSystem(self.simulation_state, self.config, self.cognitive_scaffold)
        # We replace the class with an instance for this test's purpose
        self.system_manager.register_system(lambda *args, **kwargs: mock_system_instance)

        self.assertIn(mock_system_instance, self.system_manager._systems)
        self.assertEqual(len(self.system_manager._systems), 1)

    async def test_update_all_executes_systems_concurrently(self):
        """
        Verify that the update_all method calls the update method on all
        registered systems concurrently.
        """
        # Arrange
        system1 = MockSystem(self.simulation_state, self.config, self.cognitive_scaffold)
        system2 = MockSystem(self.simulation_state, self.config, self.cognitive_scaffold)

        # To register the instances directly, we use a lambda
        self.system_manager.register_system(lambda *args, **kwargs: system1)
        self.system_manager.register_system(lambda *args, **kwargs: system2)

        current_tick = 10

        # Act
        await self.system_manager.update_all(current_tick)

        # Assert
        # Check that the async update method was awaited on each system
        system1.update.assert_awaited_once_with(current_tick=current_tick)
        system2.update.assert_awaited_once_with(current_tick=current_tick)

    async def test_get_system(self):
        """Verify that a registered system can be retrieved by its type."""

        # Arrange
        class SpecificSystem(MockSystem):
            pass

        system_instance = SpecificSystem(self.simulation_state, self.config, self.cognitive_scaffold)
        self.system_manager.register_system(lambda *args, **kwargs: system_instance)

        # Act
        retrieved_system = self.system_manager.get_system(SpecificSystem)
        nonexistent_system = self.system_manager.get_system(MockSystem)  # A different type

        # Assert
        self.assertIs(retrieved_system, system_instance)
        self.assertIsNone(nonexistent_system)


if __name__ == "__main__":
    pytest.main()
