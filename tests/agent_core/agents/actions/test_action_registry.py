# src/agent_core/tests/test_action_registry.py

from typing import Any, Dict, List

import pytest
from agent_core.agents.actions.action_interface import ActionInterface

# Subject under test
from agent_core.agents.actions.action_registry import ActionRegistry, action_registry

# Mock Objects for Testing


class MockAction(ActionInterface):
    """A valid mock action for testing registration."""

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
        return [{"param1": "value1"}]

    def execute(
        self,
        entity_id: str,
        simulation_state: Any,
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        return {"status": "success"}

    def get_feature_vector(
        self, entity_id: str, simulation_state: Any, params: Dict[str, Any]
    ) -> List[float]:
        return [1.0, 0.0]


class AnotherMockAction(ActionInterface):
    """Another valid mock action to test multiple registrations."""

    @property
    def action_id(self) -> str:
        return "another_action"

    @property
    def name(self) -> str:
        return "Another Mock Action"

    def get_base_cost(self, simulation_state: Any) -> float:
        return 2.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: Any, current_tick: int
    ) -> List[Dict[str, Any]]:
        return []

    def execute(
        self,
        entity_id: str,
        simulation_state: Any,
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        return {}

    def get_feature_vector(
        self, entity_id: str, simulation_state: Any, params: Dict[str, Any]
    ) -> List[float]:
        return [0.0, 1.0]


class InvalidAction:
    """A class that does NOT implement the ActionInterface."""

    pass


# Test Fixtures


@pytest.fixture
def fresh_registry() -> ActionRegistry:
    """Provides a fresh, empty ActionRegistry instance for each test."""
    return ActionRegistry()


# Test Cases


def test_register_valid_action(fresh_registry: ActionRegistry):
    """
    Tests that a valid action class that implements ActionInterface can be registered successfully.
    """
    # Act
    fresh_registry.register(MockAction)

    # Assert
    assert "mock_action" in fresh_registry.action_ids
    retrieved_action = fresh_registry.get_action("mock_action")
    assert retrieved_action == MockAction


def test_register_invalid_action_raises_type_error(fresh_registry: ActionRegistry):
    """
    Tests that attempting to register a class that does not implement
    ActionInterface raises a TypeError.
    """
    # Act & Assert
    with pytest.raises(
        TypeError, match="Class InvalidAction must implement ActionInterface"
    ):
        fresh_registry.register(InvalidAction)  # type: ignore[arg-type]


def test_register_duplicate_action_id_raises_value_error(
    fresh_registry: ActionRegistry,
):
    """
    Tests that attempting to register an action with an ID that is already
    in the registry raises a ValueError.
    """
    # Arrange
    fresh_registry.register(MockAction)  # Register the first time

    # Act & Assert
    with pytest.raises(
        ValueError, match="Action with ID 'mock_action' is already registered."
    ):
        fresh_registry.register(MockAction)  # Attempt to register again


def test_get_action_success(fresh_registry: ActionRegistry):
    """
    Tests that get_action successfully retrieves a registered action class.
    """
    # Arrange
    fresh_registry.register(MockAction)

    # Act
    retrieved_action = fresh_registry.get_action("mock_action")

    # Assert
    assert retrieved_action is not None
    assert retrieved_action == MockAction


def test_get_nonexistent_action_raises_value_error(fresh_registry: ActionRegistry):
    """
    Tests that attempting to get an action that has not been registered
    raises a ValueError.
    """
    # Act & Assert
    with pytest.raises(
        ValueError, match="No action with ID 'nonexistent_action' is registered."
    ):
        fresh_registry.get_action("nonexistent_action")


def test_get_all_actions(fresh_registry: ActionRegistry):
    """
    Tests that get_all_actions returns a list of all registered action classes.
    """
    # Arrange
    fresh_registry.register(MockAction)
    fresh_registry.register(AnotherMockAction)

    # Act
    all_actions = fresh_registry.get_all_actions()

    # Assert
    assert len(all_actions) == 2
    assert MockAction in all_actions
    assert AnotherMockAction in all_actions


def test_action_ids_property(fresh_registry: ActionRegistry):
    """
    Tests that the action_ids property returns a sorted list of registered action IDs.
    """
    # Arrange
    fresh_registry.register(AnotherMockAction)
    fresh_registry.register(MockAction)

    # Act
    ids = fresh_registry.action_ids

    # Assert
    assert ids == ["another_action", "mock_action"]


def test_global_singleton_instance():
    """
    Tests that the global 'action_registry' singleton instance works as expected.
    This test modifies a global state, which is generally not ideal, but it's
    necessary here to test the singleton pattern.
    """
    # Reset the global registry for a clean test environment
    action_registry._actions = {}

    # Register an action using the decorator on the global instance
    @action_registry.register
    class GlobalTestAction(MockAction):
        @property
        def action_id(self) -> str:
            return "global_test"

    # Assert that the action was registered on the global instance
    assert "global_test" in action_registry.action_ids
    retrieved_action = action_registry.get_action("global_test")
    assert retrieved_action == GlobalTestAction
