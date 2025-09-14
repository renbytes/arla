"""
Defines the Provider classes for the Tragedy of the Commons simulation.

Providers bridge the world-agnostic agent-engine and the specific rules of
this simulation. They implement interfaces from agent-core to supply
world-specific data and logic to the core cognitive systems.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import torch
from agent_core.agents.action_cost_provider_interface import ActionCostProviderInterface
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.abstractions import SimulationState as AbstractSimulationState
from agent_core.core.ecs.component import ActionPlanComponent, Component
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface
from agent_core.policy.state_encoder_interface import StateEncoderInterface
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent

from .components import EnergyComponent, PositionComponent, ResourceComponent
from .environment import CommonsEnvironment

__all__ = [
    "CommonsActionGenerator",
    "HeuristicDecisionSelector",
    "QLearningDecisionSelector",
    "CommonsStateEncoder",
    "CommonsComponentFactory",
    "CommonsRewardCalculator",
    "CommonsActionCostProvider",
]


class CommonsActionGenerator(ActionGeneratorInterface):
    """Generates possible actions for Herder agents."""

    def generate(
        self, sim_state: AbstractSimulationState, entity_id: str, tick: int
    ) -> List[ActionPlanComponent]:
        """Generates Graze, Move, and Wait actions."""
        all_plans: List[ActionPlanComponent] = []
        for action_class in action_registry.get_all_actions():
            action_instance = action_class()
            params_list = action_instance.generate_possible_params(
                entity_id, sim_state, tick
            )
            all_plans.extend(
                [
                    ActionPlanComponent(action_type=action_instance, params=p)
                    for p in params_list
                ]
            )
        return all_plans


class HeuristicDecisionSelector(DecisionSelectorInterface):
    """A simple, rule-based decision policy for baseline agents."""

    def __init__(self, simulation_state: Any = None, config: Any = None):
        """Initializes the selector."""
        pass

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        """Selects an action based on a simple survival heuristic."""
        if not possible_actions:
            return None

        # 1. If agent can graze, always graze.
        graze_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "graze"
        ]
        if graze_actions:
            return graze_actions[0]

        # 2. If not, move towards the richest adjacent grass patch.
        move_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "move"
        ]
        wait_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "wait"
        ]

        if move_actions and isinstance(sim_state.environment, CommonsEnvironment):
            env = cast(CommonsEnvironment, sim_state.environment)

            # Find the move action that goes to the patch with the most resources.
            best_move = max(
                move_actions,
                key=lambda m: env.get_resource_at(m.params["target_pos"]),
                default=None,
            )

            if best_move:
                # Get the resource amount at the best location.
                max_resource = env.get_resource_at(best_move.params["target_pos"])
                # Only move if it's better than nothing.
                if max_resource > 0:
                    return best_move

        # 3. If no beneficial move is found, wait.
        return wait_actions[0] if wait_actions else None


class QLearningDecisionSelector(DecisionSelectorInterface):
    """A decision selector that uses the agent's Q-learning network."""

    def __init__(self, simulation_state: Any, config: Any):
        self.simulation_state = simulation_state
        self.config = config
        self.state_encoder = CommonsStateEncoder()

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        """Uses epsilon-greedy strategy to select an action."""
        if not possible_actions or not isinstance(sim_state, SimulationState):
            return None

        q_comp = sim_state.get_component(entity_id, QLearningComponent)
        if not q_comp:
            return random.choice(possible_actions)
        q_comp = cast(QLearningComponent, q_comp)

        if random.random() < q_comp.current_epsilon:
            return random.choice(possible_actions)

        with torch.no_grad():
            best_action = None
            max_q_value = -float("inf")
            state_features = self.state_encoder.encode_state(
                sim_state, entity_id, self.config
            )
            state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(
                0
            )
            internal_tensor = torch.tensor([0.0], dtype=torch.float32).unsqueeze(0)

            for action_plan in possible_actions:
                if not isinstance(action_plan.action_type, ActionInterface):
                    continue
                action_features = action_plan.action_type.get_feature_vector(
                    entity_id, sim_state, action_plan.params
                )
                action_tensor = torch.tensor(
                    action_features, dtype=torch.float32
                ).unsqueeze(0)
                q_value = q_comp.utility_network(
                    state_tensor, internal_tensor, action_tensor
                ).item()
                if q_value > max_q_value:
                    max_q_value = q_value
                    best_action = action_plan
            return best_action


class CommonsStateEncoder(StateEncoderInterface):
    """Encodes the simulation state into a feature vector for the Q-Learning model."""

    def encode_state(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        config: Any,
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """Encodes local resource levels and agent's energy."""
        if not isinstance(sim_state.environment, CommonsEnvironment):
            return np.zeros(6, dtype=np.float32)

        env = cast(CommonsEnvironment, sim_state.environment)
        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        energy_comp = sim_state.get_component(entity_id, EnergyComponent)

        if not pos_comp or not energy_comp:
            return np.zeros(6, dtype=np.float32)

        pos_comp = cast(PositionComponent, pos_comp)
        energy_comp = cast(EnergyComponent, energy_comp)

        # Normalized energy level
        norm_energy = energy_comp.current_energy / energy_comp.initial_energy

        # Resource levels for cardinal directions + current location
        center = env.get_resource_at(pos_comp.position)
        north = env.get_resource_at((pos_comp.x, pos_comp.y - 1))
        south = env.get_resource_at((pos_comp.x, pos_comp.y + 1))
        east = env.get_resource_at((pos_comp.x + 1, pos_comp.y))
        west = env.get_resource_at((pos_comp.x - 1, pos_comp.y))

        max_res = sim_state.config.environment.max_resource_per_patch

        feature_vector = [
            norm_energy,
            center / max_res if max_res > 0 else 0,
            north / max_res if max_res > 0 else 0,
            south / max_res if max_res > 0 else 0,
            east / max_res if max_res > 0 else 0,
            west / max_res if max_res > 0 else 0,
        ]

        return np.array(feature_vector, dtype=np.float32)

    def encode_internal_state(
        self, components: Dict[Type[Component], Component], config: Any
    ) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class CommonsComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data."""

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        class_name = component_type.split(".")[-1]
        component_map: Dict[str, Type[Component]] = {
            "PositionComponent": PositionComponent,
            "EnergyComponent": EnergyComponent,
            "ResourceComponent": ResourceComponent,
        }
        if class_name in component_map:
            return component_map[class_name](**data)
        raise TypeError(f"Unknown component type for factory: {component_type}")


class CommonsRewardCalculator(RewardCalculatorInterface):
    """A simple reward calculator that returns the base reward."""

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type["Component"], "Component"],
    ) -> Tuple[float, Dict[str, Any]]:
        """Simply returns the base reward of the action without modification."""
        return base_reward, {"base_reward": base_reward}


class CommonsActionCostProvider(ActionCostProviderInterface):
    """Applies action costs to an agent's EnergyComponent."""

    def apply_action_cost(
        self,
        entity_id: str,
        cost: float,
        simulation_state: "AbstractSimulationState",
    ) -> None:
        """Deducts the cost from the agent's current energy."""
        energy_comp = simulation_state.get_component(entity_id, EnergyComponent)
        if energy_comp:
            energy_comp = cast(EnergyComponent, energy_comp)
            energy_comp.current_energy -= cost
