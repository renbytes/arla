"""
Unit tests for the components of the Sugarscape simulation.

These tests verify that each component initializes correctly, serializes
its data properly, and validates its internal state according to the
defined rules.
"""

import unittest

from simulations.sugarscape_sim.components import (
    CommunicationComponent,
    EnergyComponent,
    MetabolismComponent,
    PositionComponent,
)


class TestPositionComponent(unittest.TestCase):
    """Tests for the PositionComponent."""

    def test_initialization(self):
        """Verify correct initialization of coordinates."""
        pos = PositionComponent(x=10, y=20)
        self.assertEqual(pos.x, 10)
        self.assertEqual(pos.y, 20)
        self.assertEqual(pos.position, (10, 20))

    def test_to_dict(self):
        """Test serialization to a dictionary."""
        pos = PositionComponent(x=5, y=15)
        self.assertEqual(pos.to_dict(), {"x": 5, "y": 15})

    def test_validation(self):
        """Test the validation logic for coordinates."""
        pos_valid = PositionComponent(x=1, y=1)
        is_valid, errors = pos_valid.validate("agent_1")
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        pos_invalid = PositionComponent(x=1.5, y=2)  # type: ignore
        is_valid, errors = pos_invalid.validate("agent_1")
        self.assertFalse(is_valid)
        self.assertIn("Position coordinates must be integers", errors)


class TestEnergyComponent(unittest.TestCase):
    """Tests for the EnergyComponent."""

    def test_initialization(self):
        """Verify correct initialization of energy levels."""
        energy = EnergyComponent(current_energy=50.0, initial_energy=100.0)
        self.assertEqual(energy.current_energy, 50.0)
        self.assertEqual(energy.initial_energy, 100.0)

    def test_validation(self):
        """Test that energy cannot be negative."""
        energy_valid = EnergyComponent(current_energy=0.0, initial_energy=100.0)
        self.assertTrue(energy_valid.validate("agent_1")[0])

        energy_invalid = EnergyComponent(current_energy=-10.0, initial_energy=100.0)
        is_valid, errors = energy_invalid.validate("agent_1")
        self.assertFalse(is_valid)
        self.assertIn("Energy cannot be negative.", errors)


class TestMetabolismComponent(unittest.TestCase):
    """Tests for the MetabolismComponent."""

    def test_initialization(self):
        """Verify correct initialization of metabolic traits."""
        metabolism = MetabolismComponent(metabolic_rate=2, vision_range=5)
        self.assertEqual(metabolism.metabolic_rate, 2)
        self.assertEqual(metabolism.vision_range, 5)

    def test_validation(self):
        """Test validation for metabolic rate and vision range."""
        metabolism_valid = MetabolismComponent(metabolic_rate=1, vision_range=0)
        self.assertTrue(metabolism_valid.validate("agent_1")[0])

        metabolism_invalid_rate = MetabolismComponent(metabolic_rate=0, vision_range=5)
        is_valid, errors = metabolism_invalid_rate.validate("agent_1")
        self.assertFalse(is_valid)
        self.assertIn("Metabolic rate must be positive.", errors)

        metabolism_invalid_vision = MetabolismComponent(
            metabolic_rate=1, vision_range=-1
        )
        is_valid, errors = metabolism_invalid_vision.validate("agent_1")
        self.assertFalse(is_valid)
        self.assertIn("Vision range cannot be negative.", errors)


class TestCommunicationComponent(unittest.TestCase):
    """Tests for the CommunicationComponent."""

    def test_message_queue(self):
        """Test adding and retrieving messages from the queue."""
        comm = CommunicationComponent(message_range=7)
        self.assertEqual(len(comm.message_queue), 0)

        comm.add_message("agent_1", "Hello!", 10)
        self.assertEqual(len(comm.message_queue), 1)

        messages = comm.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["from"], "agent_1")
        self.assertEqual(len(comm.message_queue), 0)  # Queue should be cleared
