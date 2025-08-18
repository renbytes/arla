# tests/agent-core/core/ecs/test_components.py

from agent_core.core.ecs.component import (
    ActionOutcomeComponent,
    ActionPlanComponent,
    AffectComponent,
    BeliefSystemComponent,
    CompetenceComponent,
    EmotionComponent,
    GoalComponent,
    TimeBudgetComponent,
)


class TestTimeBudgetComponent:
    """Tests for the TimeBudgetComponent."""

    def test_validation_success(self):
        """Test that a valid component passes validation."""
        comp = TimeBudgetComponent(initial_time_budget=100.0)
        is_valid, errors = comp.validate("agent_1")
        assert is_valid
        assert not errors

    def test_validation_failure_negative_budget(self):
        """Test that a negative current budget fails validation."""
        comp = TimeBudgetComponent(initial_time_budget=100.0)
        comp.current_time_budget = -10.0
        is_valid, errors = comp.validate("agent_1")
        assert not is_valid
        assert "current_time_budget cannot be negative" in errors[0]

    def test_validation_failure_mismatched_active_state(self):
        """Test validation failure when is_active contradicts the budget."""
        comp = TimeBudgetComponent(initial_time_budget=100.0)
        comp.current_time_budget = 0
        comp.is_active = True  # Should be inactive
        is_valid, errors = comp.validate("agent_1")
        assert not is_valid
        assert "Entity marked active but has no time budget" in errors[0]

    def test_auto_fix_negative_budget(self):
        """Test that auto_fix corrects a negative budget."""
        comp = TimeBudgetComponent(initial_time_budget=100.0)
        comp.current_time_budget = -5.0
        fixed = comp.auto_fix("agent_1", {})
        assert fixed
        assert comp.current_time_budget == 0.0
        assert not comp.is_active

    def test_to_dict_serialization(self):
        """Test that the component serializes to a dictionary correctly."""
        comp = TimeBudgetComponent(initial_time_budget=150.0)
        comp.current_time_budget = 120.5
        comp.is_active = True
        data = comp.to_dict()
        assert data["initial_time_budget"] == 150.0
        assert data["current_time_budget"] == 120.5
        assert data["is_active"]


class TestEmotionComponent:
    """Tests for the EmotionComponent."""

    def test_validation_out_of_bounds(self):
        """Test that valence and arousal outside their bounds fail validation."""
        comp_valence = EmotionComponent(valence=1.5)
        is_valid, errors = comp_valence.validate("agent_1")
        assert not is_valid
        assert "Valence out of bounds" in errors[0]

        comp_arousal = EmotionComponent(arousal=-0.5)
        is_valid, errors = comp_arousal.validate("agent_1")
        assert not is_valid
        assert "Arousal out of bounds" in errors[0]

    def test_to_dict_serialization(self):
        """Test correct serialization."""
        comp = EmotionComponent(valence=0.5, arousal=0.8, current_emotion_category="happy")
        data = comp.to_dict()
        assert data["valence"] == 0.5
        assert data["arousal"] == 0.8
        assert data["current_emotion_category"] == "happy"


class TestAffectComponent:
    """Tests for the AffectComponent."""

    def test_validation_nan_dissonance(self):
        """Test that NaN cognitive dissonance fails validation."""
        comp = AffectComponent(affective_buffer_maxlen=10)
        comp.cognitive_dissonance = float("nan")
        is_valid, errors = comp.validate("agent_1")
        assert not is_valid
        assert "Cognitive dissonance is not a finite number" in errors[0]


class TestCompetenceComponent:
    """Tests for the CompetenceComponent."""

    def test_auto_fix_incorrect_type(self):
        """Test that auto_fix corrects the type of action_counts."""
        comp = CompetenceComponent()
        # Intentionally set to a wrong type
        comp.action_counts = {}
        fixed = comp.auto_fix("agent_1", {})
        assert fixed
        from collections import defaultdict

        assert isinstance(comp.action_counts, defaultdict)


class TestBeliefSystemComponent:
    """Tests for the BeliefSystemComponent."""

    def test_validation_wrong_types(self):
        """Test that incorrect attribute types fail validation."""
        comp = BeliefSystemComponent()
        comp.belief_base = []  # Should be a dict
        is_valid, errors = comp.validate("agent_1")
        assert not is_valid
        assert "'belief_base' attribute must be a dictionary" in errors[0]


# You can continue adding simple test classes for other components
# to quickly boost coverage.


def test_goal_component_validation():
    """Test GoalComponent validation logic."""
    comp = GoalComponent(embedding_dim=10)
    comp.current_symbolic_goal = "explore"
    # Fails because 'explore' is not in the symbolic_goals_data
    is_valid, errors = comp.validate("agent_1")
    assert not is_valid
    assert "not in goal data" in errors[0]

    # Should pass now
    comp.symbolic_goals_data["explore"] = {}
    is_valid, errors = comp.validate("agent_1")
    assert is_valid


def test_action_plan_and_outcome_components():
    """Simple validation and serialization tests for action-related components."""
    plan = ActionPlanComponent()
    is_valid, _ = plan.validate("agent_1")
    assert is_valid
    assert isinstance(plan.to_dict(), dict)

    outcome = ActionOutcomeComponent()
    is_valid, _ = outcome.validate("agent_1")
    assert is_valid
    assert isinstance(outcome.to_dict(), dict)
