# FILE: src/agent_core/core/ecs/abstractions.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from .component import Component


class AbstractSimulationState(ABC):
    """
    The abstract interface for the simulation's state container.
    This contract ensures that any system can interact with the state
    in a predictable way without knowing the concrete implementation.
    """

    @property
    @abstractmethod
    def event_bus(self) -> Optional[Any]:
        """Provides access to the simulation's event bus."""
        raise NotImplementedError

    @abstractmethod
    def get_component(self, entity_id: str, component_type: Type["Component"]) -> Optional["Component"]:
        """Retrieves a component of a specific type for a given entity."""
        raise NotImplementedError

    @abstractmethod
    def get_entities_with_components(
        self, component_types: List[Type["Component"]]
    ) -> Dict[str, Dict[Type["Component"], "Component"]]:
        """Retrieves all entities that have a specific set of components."""
        raise NotImplementedError


SimulationState = AbstractSimulationState
