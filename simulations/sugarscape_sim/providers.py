"""
Defines the Provider classes for the Sugarscape simulation.
Providers are the bridge between the world-agnostic agent-engine and the
specific rules of this simulation. They implement the interfaces defined in
agent-core to supply world-specific data and logic to the core cognitive systems.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import torch
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.abstractions import SimulationState as AbstractSimulationState
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    PerceptionComponent,
    TimeBudgetComponent,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.environment.perception_provider_interface import (
    PerceptionProviderInterface,
)
from agent_core.environment.state_node_encoder_interface import (
    StateNodeEncoderInterface,
)
from agent_core.environment.vitality_metrics_provider_interface import (
    VitalityMetricsProviderInterface,
)
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface
from agent_core.policy.state_encoder_interface import StateEncoderInterface
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent

from .components import (
    CommunicationComponent,
    EnergyComponent,
    MetabolismComponent,
    PositionComponent,
)
from .environment import SugarscapeEnvironment


class SugarscapePerceptionProvider(PerceptionProviderInterface):
    """Provides Sugarscape-specific sensory information to agents."""

    def update_perception(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        sim_state: AbstractSimulationState,
        current_tick: int,
    ) -> None:
        """Finds all visible agents and sugar patches within vision range."""
        if not isinstance(sim_state, SimulationState):
            return

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        metabolism_comp = sim_state.get_component(entity_id, MetabolismComponent)
        perc_comp = sim_state.get_component(entity_id, PerceptionComponent)
        env = sim_state.environment

        if not all(
            [
                pos_comp,
                metabolism_comp,
                perc_comp,
                isinstance(env, SugarscapeEnvironment),
            ]
        ):
            return

        pos_comp = cast(PositionComponent, pos_comp)
        metabolism_comp = cast(MetabolismComponent, metabolism_comp)
        perc_comp = cast(PerceptionComponent, perc_comp)
        env = cast(
            SugarscapeEnvironment, env
        )  # Add explicit cast after isinstance check

        perc_comp.visible_entities.clear()
        vision_range = metabolism_comp.vision_range

        # Perceive other agents
        for other_id, other_pos in env.agent_positions.items():
            if other_id == entity_id:
                continue
            dist = env.distance(pos_comp.position, other_pos)
            if dist <= vision_range:
                perc_comp.visible_entities[other_id] = {
                    "type": "agent",
                    "position": other_pos,
                    "distance": dist,
                }

        # Perceive sugar patches
        for y in range(env.height):
            for x in range(env.width):
                pos = (x, y)
                if env.get_sugar_at(pos) > 0:
                    dist = env.distance(pos_comp.position, pos)
                    if dist <= vision_range:
                        patch_id = f"sugar_{x}_{y}"
                        perc_comp.visible_entities[patch_id] = {
                            "type": "sugar",
                            "position": pos,
                            "distance": dist,
                            "amount": env.get_sugar_at(pos),
                        }


class SugarscapeActionGenerator(ActionGeneratorInterface):
    """Generates all possible actions for an agent in the Sugarscape."""

    def generate(
        self, sim_state: AbstractSimulationState, entity_id: str, tick: int
    ) -> List[ActionPlanComponent]:
        """Iterates through all registered actions and generates their parameters."""
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

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        """Selects an action based on a simple survival heuristic."""
        if not possible_actions or not isinstance(sim_state, SimulationState):
            return None

        harvest_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "harvest"
        ]
        if harvest_actions:
            return harvest_actions[0]

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        perc_comp = sim_state.get_component(entity_id, PerceptionComponent)
        env = sim_state.environment

        if not pos_comp or not perc_comp or not isinstance(env, SugarscapeEnvironment):
            return random.choice(possible_actions) if possible_actions else None

        pos_comp = cast(PositionComponent, pos_comp)
        perc_comp = cast(PerceptionComponent, perc_comp)

        richest_patch = None
        max_sugar = -1
        for entity_data in perc_comp.visible_entities.values():
            if entity_data["type"] == "sugar" and entity_data["amount"] > max_sugar:
                max_sugar = entity_data["amount"]
                richest_patch = entity_data["position"]

        if richest_patch:
            move_actions = [
                a
                for a in possible_actions
                if a.action_type and a.action_type.action_id == "move"
            ]
            if move_actions:
                best_move = min(
                    move_actions,
                    key=lambda m: env.distance(m.params["target_pos"], richest_patch),
                )
                return best_move

        return random.choice(possible_actions)


class SugarscapeStateEncoder(StateEncoderInterface):
    """Encodes the Sugarscape state for the Q-Learning model."""

    def encode_state(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        config: Any,
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """Encodes agent vitals and perception into a feature vector."""
        if not isinstance(sim_state, SimulationState):
            return np.zeros(10, dtype=np.float32)

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        energy_comp = sim_state.get_component(entity_id, EnergyComponent)
        perc_comp = sim_state.get_component(entity_id, PerceptionComponent)
        metabolism_comp = sim_state.get_component(entity_id, MetabolismComponent)
        env = sim_state.environment

        if not all(
            [
                pos_comp,
                energy_comp,
                perc_comp,
                metabolism_comp,
                isinstance(env, SugarscapeEnvironment),
            ]
        ):
            return np.zeros(10, dtype=np.float32)

        pos_comp = cast(PositionComponent, pos_comp)
        energy_comp = cast(EnergyComponent, energy_comp)
        perc_comp = cast(PerceptionComponent, perc_comp)
        metabolism_comp = cast(MetabolismComponent, metabolism_comp)
        env = cast(
            SugarscapeEnvironment, env
        )  # Add explicit cast after isinstance check

        agent_state = [
            pos_comp.x / env.width,
            pos_comp.y / env.height,
            energy_comp.current_energy / energy_comp.initial_energy,
        ]

        sugar_patches = sorted(
            [v for v in perc_comp.visible_entities.values() if v["type"] == "sugar"],
            key=lambda p: p["distance"],
        )

        perception_vector = []
        for i in range(2):
            if i < len(sugar_patches):
                patch = sugar_patches[i]
                dist = patch["distance"] / metabolism_comp.vision_range
                amount = patch["amount"] / env.max_sugar_per_cell
                dx = patch["position"][0] - pos_comp.x
                dy = patch["position"][1] - pos_comp.y
                angle = math.atan2(dy, dx) / math.pi
                perception_vector.extend([dist, angle, amount])
            else:
                perception_vector.extend([1.0, 0.0, 0.0])

        return np.array(agent_state + perception_vector, dtype=np.float32)

    def encode_internal_state(
        self, components: Dict[Type[Component], Component], config: Any
    ) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class SugarscapeRewardCalculator(RewardCalculatorInterface):
    """Calculates final, subjective rewards for the Sugarscape simulation."""

    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: Dict[str, Any],
        entity_components: Dict[Type[Component], "Component"],
    ) -> Tuple[float, Dict[str, Any]]:
        energy_comp = entity_components.get(EnergyComponent)
        if not energy_comp:
            return base_reward, {"base_reward": base_reward}

        energy_comp = cast(EnergyComponent, energy_comp)

        survival_penalty = 0.0
        if energy_comp.current_energy < 20:
            survival_penalty = -10.0

        final_reward = base_reward + survival_penalty
        return final_reward, {
            "base_reward": base_reward,
            "survival_penalty": survival_penalty,
        }


class SugarscapeComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data."""

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        class_name = component_type.split(".")[-1]
        component_map = {
            "PositionComponent": PositionComponent,
            "EnergyComponent": EnergyComponent,
            "MetabolismComponent": MetabolismComponent,
            "CommunicationComponent": CommunicationComponent,
            "TimeBudgetComponent": TimeBudgetComponent,
            "PerceptionComponent": PerceptionComponent,
            "QLearningComponent": QLearningComponent,
        }

        if class_name in component_map:
            if class_name == "QLearningComponent":
                return QLearningComponent(
                    state_feature_dim=9,
                    internal_state_dim=1,
                    action_feature_dim=6,
                    q_learning_alpha=0.1,
                    device=torch.device("cpu"),
                )
            return component_map[class_name](**data)
        raise TypeError(f"Unknown component type for factory: {component_type}")


class SugarscapeVitalityMetricsProvider(VitalityMetricsProviderInterface):
    """Provides normalized vitality metrics for the AffectSystem."""

    def get_normalized_vitality_metrics(
        self,
        entity_id: str,
        components: Dict[Type[Component], "Component"],
        config: Any,
    ) -> Dict[str, float]:
        energy_comp = components.get(EnergyComponent)
        if energy_comp:
            energy_comp = cast(EnergyComponent, energy_comp)
            return {
                "energy_norm": energy_comp.current_energy / energy_comp.initial_energy
            }
        return {"energy_norm": 0.5}


class SugarscapeStateNodeEncoder(StateNodeEncoderInterface):
    """Encodes the agent's state into a symbolic tuple for the CausalGraphSystem."""

    def __init__(self, simulation_state: SimulationState):
        self.simulation_state = simulation_state

    def encode_state_for_causal_graph(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        current_tick: int,
        config: Any,
    ) -> Tuple[Any, ...]:
        energy_comp = components.get(EnergyComponent)
        pos_comp = components.get(PositionComponent)
        env = self.simulation_state.environment

        energy_status = "stable"
        if energy_comp:
            energy_comp = cast(EnergyComponent, energy_comp)
            energy_ratio = energy_comp.current_energy / energy_comp.initial_energy
            if energy_ratio < 0.2:
                energy_status = "critical"
            elif energy_ratio < 0.5:
                energy_status = "low"

        local_sugar_status = "barren"
        if pos_comp and isinstance(env, SugarscapeEnvironment):
            pos_comp = cast(PositionComponent, pos_comp)
            env = cast(
                SugarscapeEnvironment, env
            )  # Add explicit cast after isinstance check
            sugar_level = env.get_sugar_at(pos_comp.position)
            if sugar_level > env.max_sugar_per_cell / 2:
                local_sugar_status = "abundant"
            elif sugar_level > 0:
                local_sugar_status = "present"

        return ("STATE", f"energy_{energy_status}", f"sugar_{local_sugar_status}")
