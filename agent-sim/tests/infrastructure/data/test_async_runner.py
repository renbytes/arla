# src/agent_sim/infrastructure/data/tests/test_async_runner.py
"""
Comprehensive unit tests for the AsyncRunner class.
"""

import asyncio
import threading
from concurrent.futures import TimeoutError
from typing import Any, Coroutine, Iterator
from unittest.mock import MagicMock, patch

import pytest

# Subject under test
from src.agent_sim.infrastructure.data.async_runner import AsyncRunner

# --- Test Coroutines ---


async def simple_coro(result: Any, delay: float = 0) -> Any:
    """A simple coroutine that optionally delays and returns a value."""
    if delay > 0:
        await asyncio.sleep(delay)
    return result


async def exception_coro() -> None:
    """A coroutine that always raises a specific exception."""
    raise ValueError("Test Exception")


# --- Fixtures ---


@pytest.fixture
def runner() -> Iterator[AsyncRunner]:
    """Provides a fresh AsyncRunner instance for each test and ensures it's closed."""
    ar = AsyncRunner()
    yield ar
    ar.close()


# --- Test Cases ---


def test_initialization(runner: AsyncRunner):
    """
    Tests that the AsyncRunner initializes its loop and thread correctly.
    """
    assert runner._loop is not None
    assert runner._thread is not None
    assert runner._thread.is_alive()
    assert runner._loop.is_running()


def test_run_async_successful_execution(runner: AsyncRunner):
    """
    Tests that run_async correctly executes a coroutine and returns its result.
    """
    expected_result = "success"
    result = runner.run_async(simple_coro(expected_result))
    assert result == expected_result


def test_run_async_propagates_exceptions(runner: AsyncRunner):
    """
    Tests that if the coroutine raises an exception, run_async re-raises it.
    """
    with pytest.raises(ValueError, match="Test Exception"):
        runner.run_async(exception_coro())


@patch("agent_sim.infrastructure.data.async_runner.asyncio.run_coroutine_threadsafe")
def test_run_async_handles_timeout(mock_run_coro: MagicMock, runner: AsyncRunner):
    """
    Tests that run_async correctly raises a TimeoutError if the future takes too long.
    This is achieved by mocking the function that creates the future.
    """
    # ARRANGE: Configure the mock future object that our patched function will return.
    # Make its result() method raise the TimeoutError we want to test.
    mock_future = MagicMock()
    mock_future.result.side_effect = TimeoutError("Test Timeout")
    mock_run_coro.return_value = mock_future

    # ACT & ASSERT
    with pytest.raises(TimeoutError, match="Test Timeout"):
        runner.run_async(simple_coro("should_timeout"))

    # Verify that result was called with the correct timeout value
    mock_future.result.assert_called_once_with(timeout=30)


def test_thread_safety_with_multiple_threads(runner: AsyncRunner):
    """
    Tests that the runner can handle concurrent submissions from multiple threads.
    """
    results = []
    num_threads = 10

    def worker(i: int):
        """Target function for each thread."""
        res = runner.run_async(simple_coro(f"result_{i}", delay=0.01))
        results.append(res)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert len(results) == num_threads
    assert set(results) == {f"result_{i}" for i in range(num_threads)}


def test_loop_restarts_if_closed(runner: AsyncRunner):
    """
    Tests that if the event loop is closed, a new one is started on the next call.
    """
    # Arrange: Get the initial loop and thread, then close the loop
    original_loop = runner._loop
    original_thread = runner._thread
    runner.close()

    # Wait for the thread to terminate
    original_thread.join(timeout=1)
    assert not original_thread.is_alive()

    # Assert that the loop is no longer running, not that it's closed.
    # The `stop()` method does not `close()` the loop.
    assert not original_loop.is_running()

    # Act: Run a new coroutine, which should trigger a loop restart
    result = runner.run_async(simple_coro("restarted"))

    # Assert
    assert result == "restarted"
    assert runner._loop is not original_loop
    assert runner._thread is not original_thread
    assert runner._loop.is_running()
    assert runner._thread.is_alive()


def test_close_is_idempotent(runner: AsyncRunner):
    """
    Tests that calling close() multiple times does not cause an error.
    """
    try:
        runner.close()
        runner.close()
    except Exception as e:
        pytest.fail(f"Calling close() multiple times raised an exception: {e}")
