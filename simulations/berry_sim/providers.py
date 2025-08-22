# simulations/berry_sim/providers.py

import numpy as np
import random
from typing import Any, Dict, List, Optional, Tuple, Type

from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    TimeBudgetComponent,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.policy.reward_calculator_interface import RewardCalculatorInterface
from agent_core.environment.state_node_encoder_interface import (
    StateNodeEncoderInterface,
)
from agent_core.environment.vitality_metrics_provider_interface import (
    VitalityMetricsProviderInterface,
)
from agent_core.policy.state_encoder_interface import StateEncoderInterface

from .actions import MoveAction, EatBerryAction
from .components import (
    HealthComponent,
    PositionComponent,
    BerryComponent,
    WaterComponent,
    RockComponent,
)
from .environment import BerryWorldEnvironment


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
    """
    An intelligent policy:
    1. If on a berry, 90% chance to eat it.
    2. If not, find the closest visible berry and move towards it.
    3. If no berries are visible, move randomly.
    """

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

        # 1. Prioritize eating if standing on a berry
        if eat_actions and random.random() < 0.9:
            return eat_actions[0]

        # 2. If not eating, find the closest berry and move towards it
        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if pos_comp and isinstance(env, BerryWorldEnvironment) and move_actions:
            vision_range = 7
            closest_berry_pos = None
            min_dist = float("inf")

            # Find the closest berry within vision range
            for berry_pos in env.berry_locations.keys():
                dist = env.distance(pos_comp.position, berry_pos)
                if dist < min_dist and dist <= vision_range:
                    min_dist = dist
                    closest_berry_pos = berry_pos

            # If a berry is visible, choose the best move towards it
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

        # 3. If no berries are visible or no better move, move randomly
        if move_actions:
            return random.choice(move_actions)

        return None


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
    """CRITICAL: Encodes world context for the CausalGraphSystem."""

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
    """Encodes state for the Q-Learning system."""

    def __init__(self, simulation_state: Any):
        self.simulation_state = simulation_state

    def encode_state(
        self,
        simulation_state: Any,
        entity_id: str,
        config: Any,
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        pos_comp = simulation_state.get_component(entity_id, PositionComponent)
        health_comp = simulation_state.get_component(entity_id, HealthComponent)

        env_params = config.environment.get("params", {})
        width = env_params.get("width", 50)
        height = env_params.get("height", 50)

        x = pos_comp.x / width if pos_comp else 0.5
        y = pos_comp.y / height if pos_comp else 0.5
        health = (
            health_comp.current_health / health_comp.initial_health
            if health_comp
            else 0.5
        )

        return np.array([x, y, health], dtype=np.float32)

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
        }

        if class_name in component_map:
            return component_map[class_name](**data)

        raise TypeError(f"Unknown component type for factory: {component_type}")
