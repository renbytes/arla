# src/agent_engine/simulation/system.py

import threading
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type, cast

# Imports from the core library
from agent_core.cognition.scaffolding import CognitiveScaffold
from agent_core.core.ecs.base import (
    CognitiveComponent,
)
from agent_core.core.ecs.abstractions import (
    AbstractSimulationState,
)
from agent_engine.simulation.abstractions import (
    CognitiveSystem,
)
from agent_engine.simulation.simulation_state import (
    SimulationState as ConcreteSimulationState,
)


class System(CognitiveSystem):
    """
    Abstract base class for all systems in the engine.
    """

    REQUIRED_COMPONENTS: Sequence[Type[CognitiveComponent]] = []

    def __init__(
        self,
        simulation_state: "AbstractSimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ) -> None:
        super().__init__(simulation_state, config, cognitive_scaffold)
        # MODIFIED: Cast self.simulation_state to the concrete type.
        # This makes the concrete methods available to all system subclasses.
        self.simulation_state: ConcreteSimulationState = cast(ConcreteSimulationState, simulation_state)

    @abstractmethod
    def update(self, current_tick: int) -> None:  # Added return type annotation
        """
        Processes the system's logic for the current simulation tick.
        The system is responsible for fetching the entities it needs to operate on
        from self.simulation_state, typically using self.REQUIRED_COMPONENTS.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # Added return type annotation
        return f"{self.__class__.__name__} System"


class SystemManager:
    """
    Manages the registration, initialization, and execution of all systems
    in a defined order.
    """

    def __init__(
        self,
        simulation_state: "AbstractSimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ) -> None:  # Added return type annotation
        self.simulation_state = simulation_state
        self.config = config
        self._systems: List[System] = []
        self._state_lock = threading.RLock()
        self.cognitive_scaffold = cognitive_scaffold

    def register_system(self, system_class: Type[System], **kwargs: Any) -> None:  # Added return type annotation
        """
        Instantiates and registers a system, passing extra keyword arguments
        to its constructor. This is crucial for dependency injection.
        """
        system_instance = system_class(self.simulation_state, self.config, self.cognitive_scaffold, **kwargs)
        self._systems.append(system_instance)
        print(f"Registered System: {system_instance}")

    def update_all(self, current_tick: int) -> None:  # Added return type annotation
        """
        Executes the update method for all registered systems in order.
        """
        with self._state_lock:
            for system in self._systems:
                try:
                    system.update(current_tick=current_tick)
                except Exception as e:
                    print(
                        f"ERROR: System '{system.__class__.__name__}' failed during update at tick {current_tick}: {e}"
                    )
                    import traceback

                    traceback.print_exc()

    def get_system(self, system_type: Type[System]) -> Optional[System]:
        """
        Retrieves a system instance of the given type.
        NOTE: This should be used sparingly, primarily for debugging or post-simulation
        analysis. Prefer using the EventBus for inter-system communication.
        """
        for system_instance in self._systems:
            if isinstance(system_instance, system_type):
                return system_instance
        print(f"Warning: System of type {system_type.__name__} not found in SystemManager.")
        return None
