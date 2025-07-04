# src/agent_core/simulation/scenario_loader_interface.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ScenarioLoaderInterface(ABC):
    """
    Abstract Base Class for a scenario loader.

    The concrete implementation of this interface will live in the final
    simulation application (e.g., agent-soul-sim). It is responsible for
    reading a scenario definition (e.g., from a JSON file) and populating
    the SimulationState with the initial set of entities, components, and
    resources.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Loads the scenario and sets up the initial state of the simulation.
        This method should populate the SimulationState with all necessary
        agents and environment objects.
        """
        raise NotImplementedError
