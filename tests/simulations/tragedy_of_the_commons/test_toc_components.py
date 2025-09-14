"""
Unit tests for the components in the Tragedy of the Commons simulation.

This file tests the data integrity, validation, and serialization methods of
each component to ensure they behave as expected.
"""

import pytest
from simulations.tragedy_of_the_commons.components import (
    EnergyComponent,
    PositionComponent,
    ResourceComponent,
)


class TestEnergyComponent:
    """Tests for the EnergyComponent."""

    def test_initialization(self):
        """Test that the component initializes with correct values."""
        comp = EnergyComponent(current_energy=80.0, initial_energy=100.0)
        assert comp.current_energy == 80.0
        assert comp.initial_energy == 100.0

    def test_to_dict_serialization(self):
        """Test that the to_dict method serializes data correctly."""
        comp = EnergyComponent(current_energy=75.5, initial_energy=100.0)
        expected_dict = {
            "current_energy": 75.5,
            "initial_energy": 100.0,
        }
        assert comp.to_dict() == expected_dict

    def test_validation_succeeds_for_valid_state(self):
        """Test that validation passes for non-negative energy."""
        comp = EnergyComponent(current_energy=0.0, initial_energy=100.0)
        is_valid, errors = comp.validate("agent_1")
        assert is_valid
        assert not errors

    def test_validation_fails_for_negative_energy(self):
        """Test that validation fails when current_energy is negative."""
        comp = EnergyComponent(current_energy=-10.0, initial_energy=100.0)
        is_valid, errors = comp.validate("agent_1")
        assert not is_valid
        assert "Energy cannot be negative" in errors[0]


class TestResourceComponent:
    """Tests for the ResourceComponent."""

    @pytest.fixture
    def resource_comp(self) -> ResourceComponent:
        """Provides a standard ResourceComponent for tests."""
        return ResourceComponent(
            current_resource=10.0, max_resource=20, regeneration_rate=0.5
        )

    def test_initialization(self, resource_comp: ResourceComponent):
        """Test that the component initializes correctly."""
        assert resource_comp.current_resource == 10.0
        assert resource_comp.max_resource == 20
        assert resource_comp.regeneration_rate == 0.5
        assert not resource_comp.is_depleted

    def test_consume_less_than_available(self, resource_comp: ResourceComponent):
        """Test consuming an amount smaller than the current resource."""
        consumed = resource_comp.consume(5.0)
        assert consumed == 5.0
        assert resource_comp.current_resource == 5.0
        assert not resource_comp.is_depleted

    def test_consume_more_than_available(self, resource_comp: ResourceComponent):
        """Test consuming an amount larger than the current resource."""
        consumed = resource_comp.consume(15.0)
        assert consumed == 10.0
        assert resource_comp.current_resource == 0.0
        assert resource_comp.is_depleted

    def test_regenerate_below_max(self, resource_comp: ResourceComponent):
        """Test resource regeneration when below the maximum."""
        resource_comp.regenerate()
        assert resource_comp.current_resource == 10.5

    def test_regenerate_at_max(self, resource_comp: ResourceComponent):
        """Test that regeneration does not exceed the maximum resource level."""
        resource_comp.current_resource = 20.0
        resource_comp.regenerate()
        assert resource_comp.current_resource == 20.0

    def test_regenerate_near_max(self, resource_comp: ResourceComponent):
        """Test that regeneration caps at the maximum resource level."""
        resource_comp.current_resource = 19.8
        resource_comp.regenerate()
        assert resource_comp.current_resource == 20.0

    def test_validation_fails_for_negative_resource(self):
        """Test that validation fails for negative resource values."""
        comp = ResourceComponent(
            current_resource=-5.0, max_resource=20, regeneration_rate=0.5
        )
        is_valid, errors = comp.validate("grass_1")
        assert not is_valid
        assert "Resource level cannot be negative" in errors[0]


class TestPositionComponent:
    """Tests for the PositionComponent."""

    def test_initialization_and_position_property(self):
        """Test component initialization and the .position property."""
        comp = PositionComponent(x=10, y=15)
        assert comp.x == 10
        assert comp.y == 15
        assert comp.position == (10, 15)

    def test_to_dict_serialization(self):
        """Test that to_dict serializes data correctly."""
        comp = PositionComponent(x=5, y=8)
        assert comp.to_dict() == {"x": 5, "y": 8}

    def test_validation_succeeds_for_integers(self):
        """Test that validation passes for integer coordinates."""
        comp = PositionComponent(x=0, y=0)
        is_valid, errors = comp.validate("agent_1")
        assert is_valid
        assert not errors

    def test_validation_fails_for_non_integers(self):
        """Test that validation fails for non-integer coordinates."""
        comp = PositionComponent(x=1.5, y=2)  # type: ignore
        is_valid, errors = comp.validate("agent_1")
        assert not is_valid
        assert "Position coordinates must be integers" in errors[0]
