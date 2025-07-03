# src/agent_engine/simulation/abstractions.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING, Type

from agent_core.core.ecs.base import CognitiveComponent

if TYPE_CHECKING:
    from agent_core.cognition.scaffolding import CognitiveScaffold
    from agent_core.core.ecs.event_bus import EventBus

    class SimulationState:
        event_bus: Optional[EventBus]
        config: Dict[str, Any]


class CognitiveSystem(ABC):
    """
    Abstract Base Class for all logic systems in the ARLA platform.
    """

    REQUIRED_COMPONENTS: Sequence[Type[CognitiveComponent]] = []

    def __init__(
        self,
        simulation_state: "SimulationState",
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ):
        self.simulation_state = simulation_state
        self.config = config
        self.cognitive_scaffold = cognitive_scaffold
        self.event_bus = simulation_state.event_bus

    @abstractmethod
    def update(self, current_tick: int) -> None:
        """
        The main logic loop for the system, called once per simulation tick.
        """
        pass
