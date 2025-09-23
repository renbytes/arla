# simulations/disease_sim/providers.py
"""
Defines the Provider classes for the Disease simulation.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Tuple,
)  # FIX: Added 'Tuple' to the import list
import networkx as nx
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.component import ActionPlanComponent, Component
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface
from agent_core.agents.action_cost_provider_interface import ActionCostProviderInterface
from agent_core.core.ecs.abstractions import SimulationState
from .actions import WaitAction
from .components import (
    DiseaseStateComponent,
    DiseaseParametersComponent,
    NeighborhoodComponent,
    PositionComponent,
    SocialNetworkComponent,
)


# --- The Interface for Social Contact ---
# In a real scenario, this would be in agent-core. For this example, we define it here.
class SocialContactProviderInterface:
    def get_contacts(self, entity_id: str) -> List[str]:
        raise NotImplementedError

    def build_network(self) -> None:
        raise NotImplementedError


# --- The Concrete Implementation ---
class SmallWorldContactProvider(SocialContactProviderInterface):
    """
    Generates and provides contacts based on a small-world network model.
    """

    def __init__(self, simulation_state: SimulationState, config: Any):
        self.simulation_state = simulation_state
        self.config = config
        self.graph = nx.Graph()

    def build_network(self):
        """Builds a Watts-Strogatz small-world graph."""
        agents = self.simulation_state.get_entities_with_components([PositionComponent])
        agent_ids = list(agents.keys())
        self.graph.add_nodes_from(agent_ids)

        # Watts-Strogatz model for small-world networks
        k = self.config.network.avg_degree
        p = self.config.network.rewiring_prob
        seed = self.config.simulation.random_seed

        sw_graph = nx.watts_strogatz_graph(n=len(agent_ids), k=k, p=p, seed=seed)

        # Map graph nodes back to our agent IDs
        id_map = {i: agent_id for i, agent_id in enumerate(agent_ids)}
        for u, v in sw_graph.edges():
            self.graph.add_edge(id_map[u], id_map[v])

        # Store contacts in components for potential later use
        for agent_id in agent_ids:
            social_comp = self.simulation_state.get_component(
                agent_id, SocialNetworkComponent
            )
            if social_comp:
                contacts = list(self.graph.neighbors(agent_id))
                # This is a simplification; a real model might distinguish short/long range
                social_comp.short_range_contacts = contacts

    def get_contacts(self, entity_id: str) -> List[str]:
        if self.graph.has_node(entity_id):
            return list(self.graph.neighbors(entity_id))
        return []


# --- Standard Boilerplate Providers ---


class DiseaseActionGenerator(ActionGeneratorInterface):
    """Generates the only possible action: Wait."""

    def generate(
        self, sim_state: SimulationState, entity_id: str, tick: int
    ) -> List[ActionPlanComponent]:
        wait_action = WaitAction()
        params = wait_action.generate_possible_params(entity_id, sim_state, tick)
        return [ActionPlanComponent(action_type=wait_action, params=p) for p in params]


class PassiveDecisionSelector(DecisionSelectorInterface):
    """A simple policy: if an agent can wait, it will."""

    def select(
        self,
        sim_state: SimulationState,
        entity_id: str,
        actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        return actions[0] if actions else None


class DiseaseComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data."""

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        class_name = component_type.split(".")[-1]
        component_map: Dict[str, Type[Component]] = {
            "PositionComponent": PositionComponent,
            "DiseaseStateComponent": DiseaseStateComponent,
            "NeighborhoodComponent": NeighborhoodComponent,
            "SocialNetworkComponent": SocialNetworkComponent,
            "DiseaseParametersComponent": DiseaseParametersComponent,
        }
        if class_name in component_map:
            return component_map[class_name](**data)
        raise TypeError(f"Unknown component type for factory: {component_type}")


class DiseaseRewardCalculator(RewardCalculatorInterface):
    """A simple reward calculator that returns the base reward."""

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type["Component"], "Component"],
    ) -> Tuple[float, Dict[str, Any]]:
        return base_reward, {"base_reward": base_reward}


class DiseaseActionCostProvider(ActionCostProviderInterface):
    """Applies action costs (none in this model)."""

    def apply_action_cost(
        self, entity_id: str, cost: float, simulation_state: "SimulationState"
    ) -> None:
        pass
