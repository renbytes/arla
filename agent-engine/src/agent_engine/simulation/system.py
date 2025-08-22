# src/agent_engine/simulation/system.py

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from agent_concurrent import ParallelSystemRunner
from agent_concurrent.runners import SystemProtocol, SystemRunner

from agent_core.cognition.scaffolding import CognitiveScaffold

from agent_engine.simulation.abstractions import CognitiveSystem
from agent_engine.simulation.simulation_state import SimulationState


class System(CognitiveSystem, SystemProtocol):
    """
    Abstract base class for all systems in the engine.
    It now directly inherits the synchronous contract from CognitiveSystem.
    """

    # No need to override __init__ as the parent class handles it.

    @abstractmethod
    def update(self, current_tick: int) -> None:
        """
        Processes the system's logic for the current simulation tick.
        The system is responsible for fetching the entities it needs to operate on
        from self.simulation_state, typically using self.REQUIRED_COMPONENTS.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} System"


class SystemManager:
    """
    Manages the registration, initialization, and execution of all systems
    using a specified runner (e.g., ParallelSystemRunner).
    """

    def __init__(
        self,
        simulation_state: "SimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ) -> None:
        self.simulation_state = simulation_state
        self.config = config
        self._systems: List[System] = []
        self.cognitive_scaffold = cognitive_scaffold

        self.runner: SystemRunner = ParallelSystemRunner()

    def register_system(self, system_class: Type[System], **kwargs: Any) -> None:
        """
        Instantiates and registers a system, passing extra keyword arguments
        to its constructor. This is crucial for dependency injection.
        """
        system_instance = system_class(
            self.simulation_state, self.config, self.cognitive_scaffold, **kwargs
        )
        self._systems.append(system_instance)
        print(f"Registered System: {system_instance}")

    # The update method is now synchronous and delegates execution to the runner.
    def update_all(self, current_tick: int) -> None:
        """
        Executes the update method for all registered systems using the runner.
        """
        # The runner handles parallel execution and error logging.
        self.runner.run(self._systems, current_tick=current_tick)

    def get_system(self, system_type: Type[System]) -> Optional[System]:
        """
        Retrieves a system instance of the given type.
        NOTE: This should be used sparingly, primarily for debugging or post-simulation
        analysis. Prefer using the EventBus for inter-system communication.
        """
        for system_instance in self._systems:
            if isinstance(system_instance, system_type):
                return system_instance
        print(
            f"Warning: System of type {system_type.__name__} not found in SystemManager."
        )
        return None
