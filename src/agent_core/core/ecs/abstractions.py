# src/agent_core/core/ecs/abstractions.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type


class CognitiveComponent(ABC):
    """
    Abstract Base Class for all data components in the ARLA platform.
    A Component is a pure data container. It should not contain any logic.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component's current state into a dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        """
        Validates the internal consistency of the component's data.
        """
        raise NotImplementedError

    def auto_fix(self, entity_id: str, config: Dict[str, Any]) -> bool:
        """
        Attempts to automatically correct any validation errors.
        """
        return False


# Forward declaration to avoid circular imports
if "TYPE_CHECKING" not in globals():
    from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent_core.cognition.scaffolding import CognitiveScaffold

    # Import EventBus so it's defined for the type hint below.
    from agent_core.core.ecs.event_bus import EventBus

    # Expand the SimulationState placeholder to include attributes
    # that other systems will need to access. This satisfies mypy.
    class SimulationState:
        event_bus: Optional[EventBus]
        entities: Dict[str, Dict[Type[CognitiveComponent], Any]]
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
    def update(self, current_tick: int):
        """
        The main logic loop for the system, called once per simulation tick.
        """
        pass
