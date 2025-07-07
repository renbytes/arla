# src/agent_concurrent/runners.py

import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, Sequence


class SystemProtocol(Protocol):
    """
    A protocol defining the required structure for a system.

    This allows the runners to operate on any object that has an async `update`
    method, without needing a direct dependency on a specific System base class
    from another library (like agent-engine).
    """

    async def update(self, current_tick: int) -> None: ...

    def __repr__(self) -> str: ...


class SystemRunner(ABC):
    """
    Abstract Base Class for all system runners.
    """

    @abstractmethod
    async def run(self, systems: Sequence[SystemProtocol], current_tick: int) -> None:
        """
        Executes the update logic for a list of systems.

        Args:
            systems: A list of system objects that conform to the SystemProtocol.
            current_tick: The current simulation tick to pass to each system's update method.
        """
        raise NotImplementedError


class SerialSystemRunner(SystemRunner):
    """
    A simple system runner that executes systems sequentially, one after another.

    This is useful for debugging or for simulations where the order of system
    execution is critical and concurrency is not desired.
    """

    async def run(self, systems: Sequence[SystemProtocol], current_tick: int) -> None:
        """
        Iterates through the systems and awaits their update method one by one.
        """
        print(f"--- Tick {current_tick}: Running {len(systems)} systems serially ---")
        for system in systems:
            try:
                await system.update(current_tick=current_tick)
            except Exception as e:
                print(f"ERROR: System '{system!r}' failed during serial update at tick {current_tick}: {e}")
                # In a serial runner, we might choose to stop or continue.
                # For robustness, we'll log and continue.
                import traceback

                traceback.print_exc()


class AsyncSystemRunner(SystemRunner):
    """
    A system runner that executes all systems concurrently using asyncio.gather.

    This is the high-performance option for simulations where systems can operate
    independently without strict ordering dependencies.
    """

    async def run(self, systems: Sequence[SystemProtocol], current_tick: int) -> None:
        """
        Creates a task for each system's update method and runs them concurrently.
        """
        print(f"--- Tick {current_tick}: Running {len(systems)} systems concurrently ---")
        tasks = [asyncio.create_task(system.update(current_tick=current_tick)) for system in systems]

        # Use asyncio.gather with return_exceptions=True to ensure that one
        # failing system does not stop the others.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # After all tasks have completed, check for and log any exceptions.
        for result, system in zip(results, systems):
            if isinstance(result, Exception):
                print(f"ERROR: System '{system!r}' failed during concurrent update at tick {current_tick}: {result}")
                import traceback

                traceback.print_exception(type(result), result, result.__traceback__)
