# agent-core/src/agent_core/core/ecs/event_bus.py

import asyncio
import inspect
import traceback
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Union, cast

# Update the EventHandler type to accept both sync and async functions.
EventHandler = Union[
    Callable[[Dict[str, Any]], None],
    Callable[[Dict[str, Any]], Coroutine[Any, Any, None]],
]


def _handle_task_exception(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception:
        print("--- ERROR IN ASYNC EVENT HANDLER ---")
        traceback.print_exc()
        print("------------------------------------")


class EventBus:
    """A simple event bus for decoupling communication between systems."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.debug_logging = config.get("enable_debug_logging", False)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribes a handler function to an event type."""
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publishes an event to all subscribed handlers."""
        if self.debug_logging:
            print(f"DEBUG: Publishing event '{event_type}'")

        for handler in self._subscribers[event_type]:
            try:
                if inspect.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(event_data))
                    task.add_done_callback(_handle_task_exception)
                else:
                    # We need to cast here because mypy can't infer that
                    # if it's not a coroutine, it must be the sync callable.
                    sync_handler = cast(Callable[[Dict[str, Any]], None], handler)
                    sync_handler(event_data)
            except Exception as e:
                print(f"ERROR: Handler {getattr(handler, '__name__', 'unknown')} failed for event '{event_type}': {e}")
                traceback.print_exc()
