# src/agent_core/core/ecs/abstractions.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Type

# Forward declaration to avoid circular imports
if "TYPE_CHECKING" not in globals():
    from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.agent_core.cognition.scaffolding import CognitiveScaffold
    from src.agent_core.core.ecs.component import SimulationState


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
