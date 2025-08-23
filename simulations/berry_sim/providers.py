# FILE: simulations/berry_sim/providers.py

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
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
from agent_engine.systems.components import QLearningComponent

from .actions import EatBerryAction, MoveAction
from .components import (
    BerryComponent,
    HealthComponent,
    PositionComponent,
    RockComponent,
    WaterComponent,
)
from .environment import BerryWorldEnvironment


class BerryPerceptionProvider(PerceptionProviderInterface):
    """Provides berry-specific sensory information to agents."""

    def update_perception(
        self,
        _entity_id: str,
        components: Dict[Type[Component], Component],
        sim_state: Any,
        _current_tick: int,
    ) -> None:
        """
        Finds all berries within an agent's vision range and updates its
        PerceptionComponent.
        """
        pos_comp = components.get(PositionComponent)
        perc_comp = components.get(PerceptionComponent)
        env = sim_state.environment

        # CORRECTED: Use explicit checks to satisfy mypy that these components
        # are not None in the code block that follows.
        if not pos_comp or not perc_comp or not isinstance(env, BerryWorldEnvironment):
            return

        perc_comp.visible_entities.clear()

        for berry_pos, berry_type in env.berry_locations.items():
            dist = env.distance(pos_comp.position, berry_pos)
            if dist <= perc_comp.vision_range:
                berry_id = f"berry_{berry_pos[0]}_{berry_pos[1]}"
                perc_comp.visible_entities[berry_id] = {
                    "type": "berry",
                    "berry_type": berry_type,
                    "position": berry_pos,
                    "distance": dist,
                }


class BerryActionGenerator(ActionGeneratorInterface):
    """Generates move and eat actions for agents."""

    def generate(self, sim_state, entity_id, tick) -> List[ActionPlanComponent]:
        actions = []
        move_action = MoveAction()
        eat_action = EatBerryAction()

        move_params = move_action.generate_possible_params(entity_id, sim_state, tick)
        actions.extend(
            [
                ActionPlanComponent(action_type=move_action, params=p)
                for p in move_params
            ]
        )

        eat_params = eat_action.generate_possible_params(entity_id, sim_state, tick)
        actions.extend(
            [ActionPlanComponent(action_type=eat_action, params=p) for p in eat_params]
        )

        return actions


class BerryDecisionSelector(DecisionSelectorInterface):
    """A simple heuristic policy for the baseline agent."""

    def __init__(self, simulation_state: Any, config: Any):
        pass

    def select(
        self, sim_state, entity_id, possible_actions: List[ActionPlanComponent]
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions:
            return None

        eat_actions = [
            a for a in possible_actions if isinstance(a.action_type, EatBerryAction)
        ]
        move_actions = [
            a for a in possible_actions if isinstance(a.action_type, MoveAction)
        ]

        if eat_actions and random.random() < 0.9:
            return eat_actions[0]

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if pos_comp and isinstance(env, BerryWorldEnvironment) and move_actions:
            vision_range = 7
            closest_berry_pos = None
            min_dist = float("inf")

            for berry_pos in env.berry_locations.keys():
                dist = env.distance(pos_comp.position, berry_pos)
                if dist < min_dist and dist <= vision_range:
                    min_dist = dist
                    closest_berry_pos = berry_pos

            if closest_berry_pos:
                best_move = None
                min_dist_to_target = float("inf")
                for move in move_actions:
                    dist_to_target = env.distance(
                        move.params["target_pos"], closest_berry_pos
                    )
                    if dist_to_target < min_dist_to_target:
                        min_dist_to_target = dist_to_target
                        best_move = move
                if best_move:
                    return best_move

        if move_actions:
            return random.choice(move_actions)

        return None


class QLearningDecisionSelector(DecisionSelectorInterface):
    """A decision selector that uses the agent's Q-learning network."""

    def __init__(self, simulation_state: Any, config: Any):
        self.simulation_state = simulation_state
        self.config = config
        self.state_encoder = BerryStateEncoder(simulation_state)

    def select(
        self,
        sim_state: Any,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions:
            return None

        q_comp = sim_state.get_component(entity_id, QLearningComponent)
        if not q_comp:
            return random.choice(possible_actions)

        epsilon = self.config.learning.q_learning.get("initial_epsilon", 0.1)
        if random.random() < epsilon:
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

            entity_components = sim_state.entities.get(entity_id)
            if not entity_components:
                return random.choice(possible_actions)

            internal_state = self.state_encoder.encode_internal_state(
                entity_components, self.config
            )
            internal_tensor = torch.tensor(
                internal_state, dtype=torch.float32
            ).unsqueeze(0)

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


class BerryVitalityMetricsProvider(VitalityMetricsProviderInterface):
    """Provides normalized health for the AffectSystem."""

    def get_normalized_vitality_metrics(
        self, entity_id, components, config
    ) -> Dict[str, float]:
        health_comp = components.get(HealthComponent)
        if health_comp:
            return {
                "health_norm": health_comp.current_health / health_comp.initial_health
            }
        return {"health_norm": 0.5}


class BerryStateNodeEncoder(StateNodeEncoderInterface):
    """Encodes world context for the CausalGraphSystem."""

    def __init__(self, simulation_state: Any):
        self.simulation_state = simulation_state

    def encode_state_for_causal_graph(
        self,
        entity_id: str,
        components: Dict[Type[Component], "Component"],
        current_tick: int,
        config: Any,
    ) -> Tuple[Any, ...]:
        pos_comp = components.get(PositionComponent)
        env = self.simulation_state.environment
        if not pos_comp or not isinstance(env, BerryWorldEnvironment):
            return ("STATE", "unknown", "unknown")

        context = env.get_environmental_context(pos_comp.position)

        health_comp = components.get(HealthComponent)
        health_status = "healthy"
        if health_comp:
            health_ratio = health_comp.current_health / health_comp.initial_health
            if health_ratio < 0.3:
                health_status = "critical"
            elif health_ratio < 0.7:
                health_status = "hurt"

        return (
            "STATE",
            f"health_{health_status}",
            f"near_water_{context['near_water']}",
            f"near_rocks_{context['near_rocks']}",
        )


class BerryStateEncoder(StateEncoderInterface):
    """
    Encodes the simulation state into a feature vector for the Q-Learning model.
    """

    def __init__(self, simulation_state: Any):
        self.simulation_state = simulation_state

    def encode_state(
        self,
        sim_state: Any,
        entity_id: str,
        config: Any,
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Creates a feature vector including agent health and sensory data.
        """
        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        health_comp = sim_state.get_component(entity_id, HealthComponent)
        perc_comp = sim_state.get_component(entity_id, PerceptionComponent)

        env_params = config.environment.get("params", {})
        width = env_params.get("width", 50)
        height = env_params.get("height", 50)

        agent_x = pos_comp.x / width if pos_comp else 0.5
        agent_y = pos_comp.y / height if pos_comp else 0.5
        health = (
            health_comp.current_health / health_comp.initial_health
            if health_comp
            else 0.5
        )
        agent_state_vector = [agent_x, agent_y, health]

        nearest_berries: Dict[str, Optional[Dict[str, Any]]] = {
            "red": None,
            "blue": None,
            "yellow": None,
        }
        if perc_comp and perc_comp.visible_entities:
            for entity_data in perc_comp.visible_entities.values():
                if entity_data.get("type") == "berry":
                    b_type = entity_data["berry_type"]

                    # CORRECTED: This more explicit check is safer and satisfies mypy.
                    # It checks for None before attempting to access the 'distance' key.
                    current_nearest = nearest_berries.get(b_type)
                    if (
                        current_nearest is None
                        or entity_data["distance"] < current_nearest["distance"]
                    ):
                        nearest_berries[b_type] = entity_data

        perception_vector = []
        for berry_type in ["red", "blue", "yellow"]:
            berry_data = nearest_berries[berry_type]
            if berry_data and pos_comp and perc_comp:
                dist = berry_data["distance"] / perc_comp.vision_range
                dx = berry_data["position"][0] - pos_comp.x
                dy = berry_data["position"][1] - pos_comp.y
                angle = math.atan2(dy, dx) / math.pi
                perception_vector.extend([dist, angle])
            else:
                perception_vector.extend([1.0, 0.0])

        return np.array(agent_state_vector + perception_vector, dtype=np.float32)

    def encode_internal_state(
        self, components: Dict[Type[Component], Component], config: Any
    ) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class BerryRewardCalculator(RewardCalculatorInterface):
    def calculate_final_reward(
        self, base_reward: float, **kwargs
    ) -> Tuple[float, Dict[str, Any]]:
        return base_reward, {"base_reward": base_reward}


class BerryComponentFactory(ComponentFactoryInterface):
    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        class_name = component_type.split(".")[-1]

        component_map = {
            "PositionComponent": PositionComponent,
            "HealthComponent": HealthComponent,
            "BerryComponent": BerryComponent,
            "WaterComponent": WaterComponent,
            "RockComponent": RockComponent,
            "TimeBudgetComponent": TimeBudgetComponent,
            "QLearningComponent": QLearningComponent,
            "PerceptionComponent": PerceptionComponent,
        }

        if class_name in component_map:
            if class_name == "QLearningComponent":
                return QLearningComponent(
                    state_feature_dim=9,
                    internal_state_dim=1,
                    action_feature_dim=4,
                    q_learning_alpha=0.1,
                    device=torch.device("cpu"),
                )
            return component_map[class_name](**data)

        raise TypeError(f"Unknown component type for factory: {component_type}")
