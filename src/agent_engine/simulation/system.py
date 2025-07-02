from abc import abstractmethod
import threading
from typing import Any, Dict, List, Optional, Sequence, Type

from agent_core.cognition.scaffolding import CognitiveScaffold
from agent_core.core.ecs.abstractions import CognitiveComponent, CognitiveSystem, SimulationState


class System(CognitiveSystem):
    """Abstract base class for all systems."""

    REQUIRED_COMPONENTS: Sequence[Type[CognitiveComponent]] = []

    def __init__(
        self,
        simulation_state: "SimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

    @abstractmethod
    def update(self, current_tick: int):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__} System"


class SystemManager:
    """Manages the registration, initialization, and execution of all systems."""

    def __init__(
        self,
        simulation_state: "SimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ):
        self.simulation_state = simulation_state
        self.config = config
        self._systems: List[System] = []
        self._state_lock = threading.RLock()
        self.cognitive_scaffold = cognitive_scaffold

    def register_system(self, system_class: Type[System], **kwargs):
        """Instantiates and registers a system, passing extra kwargs."""
        system_instance = system_class(self.simulation_state, self.config, self.cognitive_scaffold, **kwargs)
        self._systems.append(system_instance)
        print(f"Registered System: {system_instance}")

    def update_all(self, current_tick: int):
        """Executes the update method for all registered systems in order."""
        with self._state_lock:
            for system in self._systems:
                system.update(current_tick=current_tick)

    def get_system(self, system_type: Type[System]) -> Optional[System]:
        """Retrieves a system instance of the given type."""
        for system_instance in self._systems:
            if isinstance(system_instance, system_type):
                return system_instance
        print(f"Warning: System of type {system_type.__name__} not found.")
        return None
