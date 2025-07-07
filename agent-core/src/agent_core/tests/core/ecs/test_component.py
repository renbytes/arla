import numpy as np
import pytest

# Subject under test
from agent_core.core.ecs.component import (
    TimeBudgetComponent,
    EmotionComponent,
    AffectComponent,
    GoalComponent,
    IdentityComponent,
    ValidationComponent,
    MemoryComponent,
    EpisodeComponent,
    BeliefSystemComponent,
    SocialMemoryComponent,
    ValueSystemComponent,
    ActionPlanComponent,
    ActionOutcomeComponent,
    Component,
    ComponentValidationError,
)

# --- Mock Objects for Testing ---


class MockMultiDomainIdentity:
    """A mock object to inject into IdentityComponent for testing."""

    def get_global_identity_embedding(self):
        return np.zeros(4)

    def get_identity_coherence(self):
        return 0.5

    def get_identity_stability(self):
        return 0.5


# --- Top-Level and Base Class Tests ---


def test_component_validation_error():
    """Tests that the custom exception formats its message correctly."""
    error = ComponentValidationError("TestComp", "agent_1", ["error 1", "another error"])
    assert "TestComp validation failed for agent_1: error 1, another error" in str(error)


def test_base_component_default_auto_fix():
    """Tests the default auto_fix method on the base Component."""

    class SimpleComponent(Component):
        def to_dict(self):
            return {}

        def validate(self, entity_id):
            return True, []

    comp = SimpleComponent()
    # The default auto_fix should do nothing and return False.
    assert comp.auto_fix("agent_1", {}) is False


# --- Component-Specific Test Classes ---


class TestMemoryComponent:
    def test_to_dict(self):
        comp = MemoryComponent()
        comp.episodic_memory = [1, 2, 3]
        d = comp.to_dict()
        assert d["episodic_memory_count"] == 3

    def test_validate(self):
        comp = MemoryComponent()
        assert comp.validate("id")[0] is True
        comp.episodic_memory = "not a list"
        assert comp.validate("id")[0] is False


class TestIdentityComponent:
    @pytest.fixture
    def comp(self):
        return IdentityComponent(multi_domain_identity=MockMultiDomainIdentity())

    def test_to_dict(self, comp):
        d = comp.to_dict()
        assert "identity_coherence" in d
        assert d["identity_stability"] == 0.5

    def test_validate_success(self, comp):
        assert comp.validate("id")[0] is True

    def test_validate_failure(self, comp):
        comp.embedding = None
        assert comp.validate("id")[0] is False


class TestValidationComponent:
    def test_to_dict(self):
        comp = ValidationComponent()
        comp.reflection_confidence_scores = {1: 0.8}
        d = comp.to_dict()
        assert d["confidence_scores"] == {1: 0.8}

    def test_validate(self):
        comp = ValidationComponent()
        assert comp.validate("id")[0] is True
        comp.reflection_confidence_scores = "bad"
        assert comp.validate("id")[0] is False

    def test_auto_fix(self):
        comp = ValidationComponent()
        comp.reflection_confidence_scores = "bad"
        assert comp.auto_fix("id", {}) is True
        assert comp.reflection_confidence_scores == {}


class TestGoalComponent:
    def test_to_dict(self):
        comp = GoalComponent(embedding_dim=4)
        comp.current_symbolic_goal = "test_goal"
        assert comp.to_dict()["current_symbolic_goal"] == "test_goal"

    def test_validate(self):
        comp = GoalComponent(embedding_dim=4)
        # Success case: No goal set
        assert comp.validate("id")[0] is True
        # Success case: Goal is in data
        comp.current_symbolic_goal = "test_goal"
        comp.symbolic_goals_data["test_goal"] = {}
        assert comp.validate("id")[0] is True
        # Failure case
        comp.current_symbolic_goal = "dangling_goal"
        assert comp.validate("id")[0] is False


class TestTimeBudgetComponent:
    def test_to_dict(self):
        comp = TimeBudgetComponent(100)
        d = comp.to_dict()
        assert d["current_time_budget"] == 100
        assert d["is_active"] is True

    def test_validate_over_max_budget(self):
        comp = TimeBudgetComponent(100)
        comp.current_time_budget = 999
        is_valid, errors = comp.validate("id")
        assert not is_valid
        assert "exceeds max" in errors[0]

    def test_auto_fix_over_max_budget(self):
        comp = TimeBudgetComponent(100)
        comp.current_time_budget = 999
        assert comp.auto_fix("id", {}) is True
        assert comp.current_time_budget == 200  # max_time_budget

    def test_auto_fix_inactive_with_budget(self):
        comp = TimeBudgetComponent(100)
        comp.is_active = False
        comp.current_time_budget = 5  # small amount
        assert comp.auto_fix("id", {}) is True
        assert comp.current_time_budget == 0

    def test_auto_fix_no_changes(self):
        comp = TimeBudgetComponent(100)
        assert comp.auto_fix("id", {}) is False


# --- Simplified Tests for Remaining Components ---


@pytest.mark.parametrize(
    "comp_class, init_args",
    [
        (EmotionComponent, []),
        (AffectComponent, [10]),
        (EpisodeComponent, []),
        (BeliefSystemComponent, []),
        (SocialMemoryComponent, [128, "cpu"]),
        (ValueSystemComponent, []),
        (ActionPlanComponent, []),
        (ActionOutcomeComponent, []),
    ],
)
def test_simple_components_to_dict_and_validate(comp_class, init_args):
    """Tests that simple components can be created and their methods run without error."""
    comp = comp_class(*init_args)
    # Ensure to_dict and validate don't crash
    assert isinstance(comp.to_dict(), dict)
    is_valid, errors = comp.validate("entity_1")
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
