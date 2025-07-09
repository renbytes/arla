# src/agent_sim/infrastructure/data/async_runner.py

import asyncio
import threading
import time
from typing import Any, Coroutine


class AsyncRunner:
    """Thread-safe async operation runner for database operations"""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._start_event_loop()

    def _start_event_loop(self):
        """Start a dedicated event loop in a separate thread."""

        def run_loop():
            """This function runs in the new thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Assign the new loop to the instance attribute so the main thread can see it.
            self._loop = loop
            loop.run_forever()

        # Nullify the loop reference before starting the new thread.
        # This makes the `while` loop below a reliable synchronization point.
        self._loop = None
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        # Wait for the new loop to be created and assigned in the other thread.
        while self._loop is None:
            time.sleep(0.01)

    def run_async(self, coro: Coroutine) -> Any:
        """Run an async coroutine and return the result."""
        # If the thread is dead or the loop isn't running, we need a new one.
        if self._thread is None or not self._thread.is_alive() or self._loop is None or not self._loop.is_running():
            self._start_event_loop()

        # This will now always use a valid, running loop.
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def close(self):
        """Clean shutdown of the async runner."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        # It's good practice to join the thread to ensure clean shutdown
        if self._thread:
            self._thread.join(timeout=1)


# Global instance
async_runner = AsyncRunner()
