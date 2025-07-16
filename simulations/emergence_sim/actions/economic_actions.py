# simulations/emergence_sim/actions/economic_actions.py
"""
Defines the primitive economic actions for the 'emergence_sim'.
These actions form the basis of a gift/debt economy, allowing agents to
give and request resources, which is fundamental to testing theories
about the origins of money and credit.
"""

from typing import TYPE_CHECKING, Any, Dict, List

# Imports from the core library, defining the base class and interfaces
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import TimeBudgetComponent

# World-specific components needed for action generation
# These would be defined in simulations/emergence_sim/components.py
# We assume an InventoryComponent and PositionComponent exist for this simulation.
from ..components import InventoryComponent, PositionComponent

if TYPE_CHECKING:
    from agent_core.core.ecs.abstractions import SimulationState


def _create_emergence_feature_vector(
    action_id: str,
    intent: Intent,
    base_cost: float,
    params: Dict[str, Any],
    param_feature_map: Dict[str, Any],
) -> List[float]:
    """
    Helper function to create a standardized action feature vector for this simulation.
    This mirrors the utility function found in other simulation packages.
    """
    action_ids = action_registry.action_ids
    action_one_hot = [1.0 if aid == action_id else 0.0 for aid in action_ids]

    intents = list(Intent)
    intent_one_hot = [1.0 if i == intent else 0.0 for i in intents]

    time_cost_feature = [base_cost / 10.0]  # Normalize cost

    # Placeholder for parameter features, can be expanded
    param_features = [0.0] * 5
    for param_name, mapping in param_feature_map.items():
        if param_name in params:
            idx, value, normalizer = mapping
            # Ensure value is numeric before division
            numeric_value = value if isinstance(value, (int, float)) else 0.0
            param_features[idx] = float(numeric_value) / float(normalizer) if normalizer != 0 else 0.0

    return action_one_hot + intent_one_hot + time_cost_feature + param_features


@action_registry.register
class GiveResourceAction(ActionInterface):
    """An agent gives a resource to another agent, creating a social obligation."""

    action_id = "give_resource"
    name = "Give Resource"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """Returns the base time budget cost for this action."""
        # A simple, low-cost social action.
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        """
        Generates parameters for giving resources to nearby agents.
        An agent might do this if it has a surplus of resources.
        """
        params_list: List[Dict[str, Any]] = []
        inv_comp = simulation_state.get_component(entity_id, InventoryComponent)
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)

        # Agent must have resources to give and be able to perceive others
        if (
            not isinstance(inv_comp, InventoryComponent)
            or inv_comp.current_resources < 1.0
            or not isinstance(pos_comp, PositionComponent)
            or not hasattr(pos_comp, "environment")
        ):
            return []

        # Find nearby agents who are active
        entity_pos = pos_comp.position
        nearby_entities = pos_comp.environment.get_entities_in_radius(entity_pos, 2)

        for other_id, _ in nearby_entities:
            if entity_id == other_id:
                continue

            other_time_comp = simulation_state.get_component(other_id, TimeBudgetComponent)
            if isinstance(other_time_comp, TimeBudgetComponent) and other_time_comp.is_active:
                params_list.append(
                    {
                        "target_agent_id": other_id,
                        "amount": 1.0,  # Give a small, fixed amount
                        "intent": Intent.COOPERATE,
                    }
                )
        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        """
        The logic for this action is handled by the SocialCreditSystem, which
        listens for the 'execute_give_resource_action' event. This method
        simply confirms the action's intent.
        """
        return {"status": "resource_given", "amount": params.get("amount", 0)}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates the feature vector for this action variant."""
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.COOPERATE),
            self.get_base_cost(simulation_state),
            params,
            {"amount": (0, params.get("amount", 1.0), 10.0)},  # Normalize amount
        )


@action_registry.register
class RequestResourceAction(ActionInterface):
    """An agent requests a resource from another agent."""

    action_id = "request_resource"
    name = "Request Resource"

    def get_base_cost(self, simulation_state: "SimulationState") -> float:
        """Returns the base time budget cost for this action."""
        return 1.0

    def generate_possible_params(
        self, entity_id: str, simulation_state: "SimulationState", current_tick: int
    ) -> List[Dict[str, Any]]:
        """
        Generates parameters for requesting resources from nearby agents.
        An agent might do this if its resources are low.
        """
        params_list: List[Dict[str, Any]] = []
        inv_comp = simulation_state.get_component(entity_id, InventoryComponent)
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)

        # Agent must be able to perceive others
        if not isinstance(pos_comp, PositionComponent) or not hasattr(pos_comp, "environment"):
            return []

        # Condition for requesting: low resources
        if isinstance(inv_comp, InventoryComponent) and inv_comp.current_resources > 5.0:
            return []

        # Find nearby agents to request from
        entity_pos = pos_comp.position
        nearby_entities = pos_comp.environment.get_entities_in_radius(entity_pos, 2)

        for other_id, _ in nearby_entities:
            if entity_id == other_id:
                continue

            other_time_comp = simulation_state.get_component(other_id, TimeBudgetComponent)
            if isinstance(other_time_comp, TimeBudgetComponent) and other_time_comp.is_active:
                params_list.append(
                    {
                        "target_agent_id": other_id,
                        "amount": 1.0,
                        "intent": Intent.COOPERATE,
                    }
                )
        return params_list

    def execute(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
        current_tick: int,
    ) -> Dict[str, Any]:
        """
        The logic for this action is handled by the SocialCreditSystem. This
        method simply confirms the action's intent.
        """
        return {"status": "resource_requested", "amount": params.get("amount", 0)}

    def get_feature_vector(
        self,
        entity_id: str,
        simulation_state: "SimulationState",
        params: Dict[str, Any],
    ) -> List[float]:
        """Generates the feature vector for this action variant."""
        return _create_emergence_feature_vector(
            self.action_id,
            params.get("intent", Intent.COOPERATE),
            self.get_base_cost(simulation_state),
            params,
            {"amount": (0, params.get("amount", 1.0), 10.0)},
        )
