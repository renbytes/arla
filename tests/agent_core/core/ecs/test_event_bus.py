# src/agent_core/tests/core/ecs/test_event_bus.py
from unittest.mock import MagicMock

import pytest

# Subject under test
from agent_core.core.ecs.event_bus import EventBus

# Test Fixtures


@pytest.fixture
def event_bus():
    """Provides a standard EventBus instance for each test."""
    return EventBus(config={})


@pytest.fixture
def debug_event_bus():
    """Provides an EventBus instance with debug logging enabled."""
    # Create a mock object that mimics the nested Pydantic config structure.
    mock_config = MagicMock()
    mock_config.simulation.enable_debug_logging = True
    return EventBus(config=mock_config)


# Test Cases


def test_subscribe_and_publish_single_handler(event_bus: EventBus):
    """
    Tests that a single subscribed handler is called when its event is published.
    """
    # Arrange
    # Create a mock function to act as the handler
    mock_handler = MagicMock()
    event_bus.subscribe("test_event", mock_handler)

    # Act
    event_data = {"key": "value"}
    event_bus.publish("test_event", event_data)

    # Assert
    # Check that the handler was called exactly once with the correct data
    mock_handler.assert_called_once_with(event_data)


def test_publish_to_multiple_handlers(event_bus: EventBus):
    """
    Tests that all subscribed handlers for an event are called on publish.
    """
    # Arrange
    handler1 = MagicMock()
    handler2 = MagicMock()
    event_bus.subscribe("multi_handler_event", handler1)
    event_bus.subscribe("multi_handler_event", handler2)

    # Act
    event_data = {"data": 123}
    event_bus.publish("multi_handler_event", event_data)

    # Assert
    handler1.assert_called_once_with(event_data)
    handler2.assert_called_once_with(event_data)


def test_handler_not_called_for_different_event(event_bus: EventBus):
    """
    Tests that a handler is not called if a different event type is published.
    """
    # Arrange
    mock_handler = MagicMock()
    event_bus.subscribe("my_event", mock_handler)

    # Act
    event_bus.publish("another_event", {"data": "-"})

    # Assert
    mock_handler.assert_not_called()


def test_publish_with_no_subscribers(event_bus: EventBus):
    """
    Tests that publishing an event with no subscribers does not raise an error.
    """
    # Act & Assert
    try:
        event_bus.publish("unsubscribed_event", {})
    except Exception as e:
        pytest.fail(
            f"Publishing to an event with no subscribers raised an exception: {e}"
        )


def test_handler_exception_does_not_stop_others(event_bus: EventBus, capsys):
    """
    Tests that if one handler raises an exception, other handlers for the same
    event are still executed.
    """

    # Arrange
    def failing_handler(data):
        raise ValueError("This handler is designed to fail")

    handler2 = MagicMock()

    event_bus.subscribe("resilient_event", failing_handler)
    event_bus.subscribe("resilient_event", handler2)

    # Act
    event_data = {"important": "data"}
    event_bus.publish("resilient_event", event_data)

    # Assert
    # The second handler should have been called despite the first one failing
    handler2.assert_called_once_with(event_data)

    # Check the combined output of stdout and stderr
    captured = capsys.readouterr()
    full_output = captured.out + captured.err

    assert (
        "ERROR: Handler failing_handler failed for event 'resilient_event'"
        in full_output
    )
    assert "ValueError: This handler is designed to fail" in full_output


def test_debug_logging_output(debug_event_bus: EventBus, capsys):
    """
    Tests that debug messages are printed when the bus is configured for it.
    """
    # Arrange
    mock_handler = MagicMock()
    # Give the mock a name for cleaner test output
    mock_handler.__name__ = "my_mock_handler"
    debug_event_bus.subscribe("debug_event", mock_handler)

    # Act
    debug_event_bus.publish("debug_event", {"info": "debug info"})

    # Assert
    captured = capsys.readouterr()
    assert "DEBUG: Publishing event 'debug_event'" in captured.out
