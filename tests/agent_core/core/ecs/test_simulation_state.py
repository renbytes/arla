# agent-core/tests/core/ecs/test_simulation_state.py

import unittest
from unittest.mock import MagicMock

from agent_core.core.ecs.component import Component
from agent_engine.simulation.simulation_state import SimulationState


class MockComponent(Component):
    """A mock component for testing purposes."""

    def to_dict(self):
        return {}

    def validate(self, entity_id: str):
        return True, []


class TestSimulationState(unittest.TestCase):
    """
    Contract tests for the SimulationState class.

    These tests verify the core functionality of the ECS state manager,
    ensuring that entities and components can be added, retrieved, and
    removed reliably.
    """

    def setUp(self):
        """Set up a new SimulationState for each test."""
        self.config = MagicMock()
        self.device = "cpu"
        self.simulation_state = SimulationState(self.config, self.device)

    def test_add_and_get_entity(self):
        """Verify that an entity can be added and its components retrieved."""
        entity_id = "agent_1"
        self.simulation_state.add_entity(entity_id)
        self.assertIn(entity_id, self.simulation_state.entities)
        self.assertEqual(self.simulation_state.entities[entity_id], {})

    def test_add_entity_that_already_exists(self):
        """Test that adding an existing entity raises a ValueError."""
        entity_id = "agent_1"
        self.simulation_state.add_entity(entity_id)
        with self.assertRaises(ValueError):
            self.simulation_state.add_entity(entity_id)

    def test_remove_entity(self):
        """Verify that an entity can be successfully removed."""
        entity_id = "agent_1"
        self.simulation_state.add_entity(entity_id)
        self.simulation_state.remove_entity(entity_id)
        self.assertNotIn(entity_id, self.simulation_state.entities)

    def test_remove_nonexistent_entity(self):
        """Test that removing a non-existent entity does not raise an error."""
        entity_id = "agent_1"
        # Should not raise an error
        self.simulation_state.remove_entity(entity_id)
        self.assertNotIn(entity_id, self.simulation_state.entities)

    def test_add_and_get_component(self):
        """Verify that a component can be added to an entity and retrieved."""
        entity_id = "agent_1"
        component = MockComponent()
        self.simulation_state.add_entity(entity_id)
        self.simulation_state.add_component(entity_id, component)

        retrieved_component = self.simulation_state.get_component(entity_id, MockComponent)
        self.assertIs(retrieved_component, component)

    def test_add_component_to_nonexistent_entity(self):
        """Test that adding a component to a non-existent entity raises a ValueError."""
        entity_id = "agent_1"
        component = MockComponent()
        with self.assertRaises(ValueError):
            self.simulation_state.add_component(entity_id, component)

    def test_get_component_from_nonexistent_entity(self):
        """Test that getting a component from a non-existent entity returns None."""
        retrieved_component = self.simulation_state.get_component("nonexistent_agent", MockComponent)
        self.assertIsNone(retrieved_component)

    def test_get_nonexistent_component_from_entity(self):
        """Test that getting a non-existent component from an entity returns None."""
        entity_id = "agent_1"
        self.simulation_state.add_entity(entity_id)
        retrieved_component = self.simulation_state.get_component(entity_id, MockComponent)
        self.assertIsNone(retrieved_component)

    def test_get_entities_with_components(self):
        """Verify retrieval of entities that have a specific set of components."""
        # Arrange
        entity_1 = "agent_1"
        entity_2 = "agent_2"
        entity_3 = "agent_3"

        comp1 = MockComponent()

        # Create another mock component type for varied testing
        class AnotherMockComponent(Component):
            def to_dict(self):
                return {}

            def validate(self, entity_id: str):
                return True, []

        comp2 = AnotherMockComponent()

        self.simulation_state.add_entity(entity_1)
        self.simulation_state.add_component(entity_1, comp1)

        self.simulation_state.add_entity(entity_2)
        self.simulation_state.add_component(entity_2, comp1)
        self.simulation_state.add_component(entity_2, comp2)

        self.simulation_state.add_entity(entity_3)
        self.simulation_state.add_component(entity_3, comp2)

        # Act
        entities_with_mock = self.simulation_state.get_entities_with_components([MockComponent])
        entities_with_both = self.simulation_state.get_entities_with_components([MockComponent, AnotherMockComponent])
        entities_with_another = self.simulation_state.get_entities_with_components([AnotherMockComponent])

        # Assert
        self.assertIn(entity_1, entities_with_mock)
        self.assertIn(entity_2, entities_with_mock)
        self.assertNotIn(entity_3, entities_with_mock)
        self.assertEqual(len(entities_with_mock), 2)

        self.assertNotIn(entity_1, entities_with_both)
        self.assertIn(entity_2, entities_with_both)
        self.assertNotIn(entity_3, entities_with_both)
        self.assertEqual(len(entities_with_both), 1)

        self.assertNotIn(entity_1, entities_with_another)
        self.assertIn(entity_2, entities_with_another)
        self.assertIn(entity_3, entities_with_another)
        self.assertEqual(len(entities_with_another), 2)


if __name__ == "__main__":
    unittest.main()
