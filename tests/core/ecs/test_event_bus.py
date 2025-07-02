# tests/core/ecs/test_event_bus.py

from unittest.mock import MagicMock
import pytest

# FIX: Correct the import path to reflect the new library structure
from agent_core.core.ecs.event_bus import EventBus


def test_subscribe_and_publish():
    """Tests that a handler can subscribe and is called on publish."""
    bus = EventBus(config={})
    handler = MagicMock()

    bus.subscribe("test_event", handler)
    bus.publish("test_event", {"data": "test_data"})

    handler.assert_called_once_with({"data": "test_data"})


def test_multiple_subscribers():
    """Tests that multiple handlers are called for the same event."""
    bus = EventBus(config={})
    handler1 = MagicMock()
    handler2 = MagicMock()

    bus.subscribe("multi_event", handler1)
    bus.subscribe("multi_event", handler2)
    bus.publish("multi_event", {"value": 42})

    handler1.assert_called_once_with({"value": 42})
    handler2.assert_called_once_with({"value": 42})


def test_no_subscribers():
    """Tests that publishing an event with no subscribers does not raise an error."""
    bus = EventBus(config={})
    try:
        bus.publish("unsubscribed_event", {})
    except Exception as e:
        pytest.fail(f"Publishing to an unsubscribed event raised an exception: {e}")


def test_handler_exception_does_not_stop_others():
    """Tests that if one handler fails, other handlers for the same event are still called."""
    bus = EventBus(config={})

    def failing_handler(data):
        raise ValueError("I failed")

    handler1 = MagicMock()
    handler3 = MagicMock()

    bus.subscribe("robust_event", handler1)
    bus.subscribe("robust_event", failing_handler)
    bus.subscribe("robust_event", handler3)

    # This should not raise an exception
    bus.publish("robust_event", {"data": "important"})

    # Check that the handlers before and after the failing one were still called
    handler1.assert_called_once_with({"data": "important"})
    handler3.assert_called_once_with({"data": "important"})
