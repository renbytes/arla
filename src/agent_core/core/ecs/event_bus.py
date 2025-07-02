# src/agent_core/core/ecs/event_bus.py

from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventBus:
    """A simple event bus for decoupling communication between systems."""

    def __init__(self, config: Dict[str, Any]):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.debug_logging = config.get("enable_debug_logging", False)

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribes a handler function to an event type."""
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publishes an event to all subscribed handlers."""
        if self.debug_logging:
            print(f"DEBUG: Publishing event '{event_type}' with data keys: {list(event_data.keys())}")

        for i, handler in enumerate(self._subscribers[event_type]):
            try:
                if self.debug_logging:
                    print(f"DEBUG: Calling handler {i}: {handler.__name__} for event '{event_type}'")
                handler(event_data)
                if self.debug_logging:
                    print(f"DEBUG: Handler {i} completed successfully")
            except Exception as e:
                print(f"ERROR: Handler {i} ({handler.__name__}) failed for event '{event_type}': {e}")
                import traceback

                traceback.print_exc()
