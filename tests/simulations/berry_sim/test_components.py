# FILE: tests/simulations/berry_sim/test_components.py
"""
Unit tests for the components defined in the berry_sim simulation.

Ensures that each component initializes correctly, handles data validation,
and serializes its state as expected.
"""

from simulations.berry_sim.components import (
    PositionComponent,
    HealthComponent,
    BerryComponent,
    WaterComponent,
    RockComponent,
)


class TestPositionComponent:
    """Tests for the PositionComponent."""

    def test_initialization(self):
        """Verify that the component initializes with correct default and custom values."""
        pos_comp_default = PositionComponent()
        assert pos_comp_default.position == (0, 0)

        pos_comp_custom = PositionComponent(x=10, y=20)
        assert pos_comp_custom.position == (10, 20)

    def test_to_dict_serialization(self):
        """Check if the component serializes to a dictionary correctly."""
        pos_comp = PositionComponent(x=5, y=15)
        expected_dict = {"x": 5, "y": 15}
        assert pos_comp.to_dict() == expected_dict

    def test_validation(self):
        """Test the validation logic for correct and incorrect data types."""
        pos_comp_valid = PositionComponent(x=1, y=2)
        is_valid, errors = pos_comp_valid.validate("agent_1")
        assert is_valid is True
        assert not errors

        pos_comp_invalid = PositionComponent(x=1.5, y=2.0)
        is_valid, errors = pos_comp_invalid.validate("agent_1")
        assert is_valid is False
        assert len(errors) == 1
        assert "integers" in errors[0]


class TestHealthComponent:
    """Tests for the HealthComponent."""

    def test_initialization(self):
        """Verify correct initialization of health values."""
        health_comp = HealthComponent(current_health=80.0, initial_health=100.0)
        assert health_comp.current_health == 80.0
        assert health_comp.initial_health == 100.0

    def test_to_dict_serialization(self):
        """Check if the component serializes to a dictionary correctly."""
        health_comp = HealthComponent(current_health=75.5, initial_health=100.0)
        expected_dict = {"current_health": 75.5, "initial_health": 100.0}
        assert health_comp.to_dict() == expected_dict

    def test_validation(self):
        """Test validation for health values."""
        health_comp_valid = HealthComponent(current_health=50.0, initial_health=100.0)
        is_valid, errors = health_comp_valid.validate("agent_1")
        assert is_valid is True
        assert not errors

        health_comp_invalid = HealthComponent(
            current_health=-10.0, initial_health=100.0
        )
        is_valid, errors = health_comp_invalid.validate("agent_1")
        assert is_valid is False
        assert "negative" in errors[0]


class TestStaticComponents:
    """Tests for simple, static components like Berry, Water, and Rock."""

    def test_berry_component(self):
        """Test the BerryComponent for initialization and validation."""
        for berry_type in ["red", "blue", "yellow"]:
            berry_comp = BerryComponent(berry_type=berry_type)
            assert berry_comp.to_dict() == {"berry_type": berry_type}
            is_valid, errors = berry_comp.validate("berry_1")
            assert is_valid is True
            assert not errors

        invalid_berry_comp = BerryComponent(berry_type="purple")
        is_valid, errors = invalid_berry_comp.validate("berry_1")
        assert is_valid is False
        assert "Invalid berry type" in errors[0]

    def test_water_component(self):
        """Test the WaterComponent."""
        water_comp = WaterComponent()
        assert water_comp.to_dict() == {}
        is_valid, errors = water_comp.validate("water_1")
        assert is_valid is True
        assert not errors

    def test_rock_component(self):
        """Test the RockComponent."""
        rock_comp = RockComponent()
        assert rock_comp.to_dict() == {}
        is_valid, errors = rock_comp.validate("rock_1")
        assert is_valid is True
        assert not errors
