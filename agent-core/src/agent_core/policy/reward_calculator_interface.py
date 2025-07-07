# src/agent_core/policy/reward_calculator_interface.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple, Type

if TYPE_CHECKING:
    from agent_core.core.ecs.component import Component


class RewardCalculatorInterface(ABC):
    """
    Abstract Base Class for a reward calculator.

    This interface allows the engine to be decoupled from the specific reward
    logic of a given simulation. The concrete implementation will be provided
    by the final application (e.g., agent-soul-sim) and will contain the
    specific bonuses, penalties, and value-based multipliers for that world's rules.
    """

    @abstractmethod
    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type["Component"], "Component"],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the final, subjective reward for an agent based on the outcome
        of an action and the agent's internal state (e.g., its values).

        Args:
            base_reward: The initial, objective reward from the action outcome.
            action_type: The type of action performed (e.g., an ActionInterface instance).
            action_intent: The intent behind the action (e.g., 'COOPERATE').
            outcome_details: A dictionary with world-specific details about the outcome.
            entity_components: A map of the acting agent's components.

        Returns:
            A tuple containing:
            - The final calculated reward (float).
            - A dictionary detailing the breakdown of bonuses and penalties for logging.
        """
        raise NotImplementedError
