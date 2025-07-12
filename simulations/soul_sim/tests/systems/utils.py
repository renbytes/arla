# tests/systems/utils.py

from typing import Any, Dict, List


class MockEventBus:
    """A mock event bus to capture published events for testing."""

    def __init__(self):
        self.published_events: List[Dict[str, Any]] = []

    def subscribe(self, event_type: str, handler):
        pass

    def publish(self, event_type: str, event_data: Dict[str, Any]):
        self.published_events.append({"type": event_type, "data": event_data})

    def get_last_published_event(self):
        return self.published_events[-1] if self.published_events else None
