# agent-core/src/agent_core/core/ecs/event_bus.py

import asyncio
import inspect
import traceback
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Set, Union, cast

# Update the EventHandler type to accept both sync and async functions.
EventHandler = Union[
    Callable[[Dict[str, Any]], None],
    Callable[[Dict[str, Any]], Coroutine[Any, Any, None]],
]


class EventBus:
    """A simple event bus for decoupling communication between systems."""

    def __init__(self, config: Any) -> None:
        """Initializes the event bus."""
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        # A set to keep track of all "fire-and-forget" async tasks.
        self._pending_tasks: Set[asyncio.Task] = set()

        self.debug_logging = (
            config.simulation.enable_debug_logging
            if hasattr(config, "simulation")
            and hasattr(config.simulation, "enable_debug_logging")
            else False
        )

    # This private method will now also remove the task from our tracking set.
    def _handle_task_exception(self, task: asyncio.Task) -> None:
        """Callback to handle exceptions in async tasks and clean them up."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass  # Ignore cancelled errors, as they are expected on shutdown.
        except Exception:
            print("--- ERROR IN ASYNC EVENT HANDLER")
            traceback.print_exc()
            print("------------------------------------")
        finally:
            # Remove the task from the pending set once it's complete.
            self._pending_tasks.discard(task)

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
                    # --- MODIFICATION START ---
                    # Create the task and add it to our tracking set.
                    task = asyncio.create_task(handler(event_data))
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._handle_task_exception)
                    # --- MODIFICATION END ---
                else:
                    sync_handler = cast(Callable[[Dict[str, Any]], None], handler)
                    sync_handler(event_data)
            except Exception as e:
                print(
                    f"ERROR: Handler {getattr(handler, '__name__', 'unknown')} failed for event '{event_type}': {e}"
                )
                traceback.print_exc()

    async def flush(self, timeout: float = 10.0) -> None:
        """
        Waits for all pending asynchronous tasks to complete.
        This should be called before shutting down the application.
        """
        if not self._pending_tasks:
            return

        print(
            f"--- Flushing Event Bus: Waiting for {len(self._pending_tasks)} background tasks..."
        )
        # Create a shielded group of tasks to wait for.
        # `asyncio.shield` can prevent the gather itself from being cancelled.
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._pending_tasks, return_exceptions=True),
                timeout=timeout,
            )
            print("--- Event Bus flushed successfully.")
        except asyncio.TimeoutError:
            print(
                f"--- WARNING: Event Bus flush timed out after {timeout} seconds. Some tasks may be lost."
            )
