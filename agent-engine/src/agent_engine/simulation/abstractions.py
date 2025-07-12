# src/agent_engine/simulation/abstractions.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type

from agent_core.core.ecs.base import CognitiveComponent

# This is the standard pattern to handle circular imports for type checking.
# The block is ignored at runtime but read by static analyzers like Pylance.
if TYPE_CHECKING:
    from agent_core.cognition.scaffolding import CognitiveScaffold
    from agent_core.core.ecs.event_bus import EventBus

    from .simulation_state import SimulationState


class CognitiveSystem(ABC):
    """
    Abstract Base Class for all logic systems in the ARLA platform.
    """

    REQUIRED_COMPONENTS: Sequence[Type[CognitiveComponent]] = []

    # Renamed the method from 'init' to the standard '__init__'.
    # This corrects the super() call chain and resolves the TypeError.
    def __init__(
        self,
        simulation_state: "SimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ):
        self.simulation_state = simulation_state
        self.config = config
        self.cognitive_scaffold = cognitive_scaffold

        # We can explicitly type self.event_bus to resolve the "not accessed" warning.
        self.event_bus: Optional["EventBus"] = simulation_state.event_bus

    @abstractmethod
    async def update(self, current_tick: int) -> None:
        """
        The main logic loop for the system, called once per simulation tick.
        """
        pass
