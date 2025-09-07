import pytest
from typing import Any, Dict, List
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import ActionRegistry, action_registry
from agent_core.agents.actions.action_outcome import ActionOutcome


# A simple, valid action for testing
@action_registry.register
class MockAction(ActionInterface):
    @property
    def action_id(self) -> str:
        return "mock_action"

    @property
    def name(self) -> str:
        return "Mock Action"

    def get_base_cost(self, simulation_state: Any) -> float:
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: Any, current_tick: int
    ) -> List[Dict[str, Any]]:
        return [{"param": 1}]

    def execute(
        self,
        entity_id: str,
        simulation_state: Any,
        params: Dict[str, Any],
        current_tick: int,
    ) -> ActionOutcome:
        return ActionOutcome(
            success=True, message="Executed mock action", base_reward=1.0
        )

    def get_feature_vector(
        self, entity_id: str, simulation_state: Any, params: Dict[str, Any]
    ) -> List[float]:
        return [1.0]


# Another action with a different ID
@action_registry.register
class AnotherMockAction(MockAction):
    @property
    def action_id(self) -> str:
        return "another_mock"

    @property
    def name(self) -> str:
        return "Another Mock"


# An invalid action that doesn't implement the interface
class NotAnAction:
    pass


# An action with a duplicate ID for testing strict mode
class DuplicateMockAction(MockAction):
    @property
    def action_id(self) -> str:
        return "mock_action"

    @property
    def name(self) -> str:
        return "Duplicate Mock"

    def execute(
        self,
        entity_id: str,
        simulation_state: Any,
        params: Dict[str, Any],
        current_tick: int,
    ) -> ActionOutcome:
        return ActionOutcome(
            success=True, message="Executed duplicate action", base_reward=2.0
        )


def test_singleton_registry():
    """Tests that the global action_registry instance works correctly."""
    assert "mock_action" in action_registry.action_ids
    assert "another_mock" in action_registry.action_ids
    action_class = action_registry.get_action("mock_action")
    assert issubclass(action_class, ActionInterface)
    assert action_class().name == "Mock Action"


def test_register_decorator():
    """Tests the registration of a valid action class."""
    registry = ActionRegistry()

    @registry.register
    class TestAction(MockAction):
        @property
        def action_id(self) -> str:
            return "test_action"

    assert "test_action" in registry.action_ids
    assert len(registry.get_all_actions()) == 1


def test_register_invalid_class_raises_type_error():
    """Tests that registering a class that doesn't implement ActionInterface raises an error."""
    registry = ActionRegistry()
    with pytest.raises(TypeError):
        registry.register(NotAnAction)


def test_get_action_success():
    """Tests retrieving a registered action."""
    registry = ActionRegistry()
    registry.register(MockAction)
    action_class = registry.get_action("mock_action")
    assert action_class == MockAction


def test_get_nonexistent_action_raises_value_error():
    """Tests that retrieving a non-existent action raises an error."""
    registry = ActionRegistry()
    with pytest.raises(ValueError):
        registry.get_action("nonexistent_action")


def test_strict_mode_prevents_duplicates():
    """Tests that strict mode raises a ValueError on duplicate action_id."""
    registry = ActionRegistry(strict=True)
    registry.register(MockAction)
    with pytest.raises(ValueError):
        registry.register(DuplicateMockAction)


def test_override_mode_allows_duplicates():
    """Tests that override mode (default) allows silent replacement of actions."""
    registry = ActionRegistry(strict=False)
    registry.register(MockAction)
    registry.register(DuplicateMockAction)

    # The new action should have replaced the old one
    action_class = registry.get_action("mock_action")
    assert issubclass(action_class, DuplicateMockAction)
    instance = action_class()
    # Check that it's the new one by checking a property or return value
    assert instance.name == "Duplicate Mock"
    assert instance.execute("", {}, {}, 0).base_reward == 2.0
