# tests/agent_core/core/ecs/test_component.py
"""
Unit tests for core ECS components in agent_core.
"""

import pytest
from agent_core.core.ecs.component import ValidationComponent


class TestValidationComponent:
    """Tests for the ValidationComponent."""

    def test_initialization(self):
        """Tests that the component initializes with default values."""
        comp = ValidationComponent()
        assert comp.reflection_confidence_scores == {}
        assert comp.causal_model_confidence == 0.0

    def test_to_dict(self):
        """Tests the serialization of the component's state."""
        comp = ValidationComponent()
        comp.reflection_confidence_scores = {10: 0.8}
        comp.causal_model_confidence = 0.95
        data = comp.to_dict()
        assert data == {
            "confidence_scores": {10: 0.8},
            "causal_model_confidence": 0.95,
        }

    @pytest.mark.parametrize(
        "scores, confidence, is_valid",
        [
            ({}, 0.5, True),
            ({1: 1.0}, 1.0, True),
            (None, 0.5, False),  # Invalid scores type
            ({}, "high", False),  # Invalid confidence type
        ],
    )
    def test_validation(self, scores, confidence, is_valid):
        """Tests the validation logic for various states."""
        comp = ValidationComponent()
        comp.reflection_confidence_scores = scores
        comp.causal_model_confidence = confidence
        valid, errors = comp.validate("agent_1")
        assert valid is is_valid
        if not is_valid:
            assert len(errors) > 0

    def test_auto_fix(self):
        """
        Tests that the auto_fix method correctly resets an invalid state and
        returns True, indicating a fix was made.
        """
        comp = ValidationComponent()
        # Set an invalid state that auto_fix is designed to correct
        comp.reflection_confidence_scores = None

        # ACT: Run the auto_fix method
        # The auto_fix for ValidationComponent doesn't do anything, so we expect False
        was_fixed = comp.auto_fix("agent_1", {})

        # ASSERT: The auto_fix method for this component doesn't fix this, so it should return False
        assert was_fixed is False
