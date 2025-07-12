# agent-core/src/agent_core/core/ecs/component_factory_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict

from .component import Component


class ComponentFactoryInterface(ABC):
    """
    An interface for a class responsible for creating component instances.
    This allows the generic restoration process to be decoupled from the
    specific constructor arguments of any given component.
    """

    @abstractmethod
    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        """
        Creates an instance of a component from its type name and saved data.

        Args:
            component_type: The full string path of the component class.
            data: The dictionary of data to pass to the component's constructor.

        Returns:
            A new component instance.
        """
        raise NotImplementedError
