# src/agent_core/core/ecs/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


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
