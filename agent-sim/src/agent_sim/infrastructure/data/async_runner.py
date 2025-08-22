# src/agent_sim/infrastructure/data/async_runner.py

import asyncio
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Coroutine


class AsyncRunner(ABC):
    """Abstract Base Class defining the interface for an async runner."""

    @abstractmethod
    def run_async(self, coro: Coroutine) -> Any:
        """Runs a coroutine from a synchronous context."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Cleans up any resources used by the runner."""
        raise NotImplementedError


class SimpleAsyncRunner(AsyncRunner):
    """
    A simple, single-threaded runner that uses asyncio.run().
    This is compatible with Celery's 'solo' pool and avoids threading issues.
    """

    def run_async(self, coro: Coroutine) -> Any:
        return asyncio.run(coro)

    def close(self) -> None:
        pass  # No-op, as asyncio.run() manages the loop lifecycle.


class ThreadedAsyncRunner(AsyncRunner):
    """
    The original multi-threaded runner that uses a dedicated background
    thread for the asyncio event loop.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._start_event_loop()

    def _start_event_loop(self) -> None:
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()

        self._loop = None
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()

        while self._loop is None:
            time.sleep(0.01)

    def run_async(self, coro: Coroutine) -> Any:
        if (
            self._thread is None
            or not self._thread.is_alive()
            or self._loop is None
            or not self._loop.is_running()
        ):
            self._start_event_loop()

        if self._loop is None:
            raise RuntimeError("AsyncRunner event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    def close(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=1)


# Use an environment variable to select the runner mode.
# Defaults to 'threaded' as it correctly manages a persistent event loop,
# which is required for libraries like SQLAlchemy's async engine.
# The 'simple' runner (using asyncio.run()) is an anti-pattern for this use case
# and causes event loop conflicts.
runner_mode = os.getenv("ASYNC_RUNNER_MODE", "threaded").lower()

# Explicitly type the variable with the base class to satisfy mypy
async_runner: AsyncRunner

if runner_mode == "threaded":
    print("INFO: Using multi-threaded AsyncRunner.")
    async_runner = ThreadedAsyncRunner()
else:
    print("INFO: Using simple, single-threaded AsyncRunner.")
    async_runner = SimpleAsyncRunner()
