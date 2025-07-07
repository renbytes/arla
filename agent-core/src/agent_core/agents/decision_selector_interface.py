# src/agent_core/agents/decision_selector_interface.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState
    from agent_core.core.ecs.component import ActionPlanComponent


class DecisionSelectorInterface(ABC):
    """
    Abstract Base Class for a decision selector.

    The concrete implementation of this interface will live in the final
    simulation application (e.g., agent-soul-sim). It is responsible for
    evaluating a list of possible actions and selecting the best one for an
    agent to execute on a given tick. This is where the core policy of the
    agent (e.g., Q-learning, rule-based, etc.) is implemented.
    """

    @abstractmethod
    def select(
        self,
        simulation_state: "SimulationState",
        entity_id: str,
        possible_actions: List["ActionPlanComponent"],
    ) -> Optional["ActionPlanComponent"]:
        """
        Selects the best action plan for an entity from a list of possibilities.

        Args:
            simulation_state: The current state of the simulation.
            entity_id: The ID of the agent making the decision.
            possible_actions: A list of valid ActionPlanComponent objects.

        Returns:
            The chosen ActionPlanComponent, or None if no action is selected.
        """
        raise NotImplementedError
