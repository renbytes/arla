# FILE: simulations/berry_sim/providers.py

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
import torch
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
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

from .actions import EatBerryAction, MoveAction
from .components import (
    BerryComponent,
    HealthComponent,
    MetabolicBoostComponent,
    PositionComponent,
    PurifierCrystalComponent,
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
        sim_state: AbstractSimulationState,
        _current_tick: int,
    ) -> None:
        """Finds all berries and crystals within vision range."""
        if not isinstance(sim_state, SimulationState):
            return

        pos_comp = components.get(PositionComponent)
        perc_comp = components.get(PerceptionComponent)
        env = sim_state.environment

        if not pos_comp or not perc_comp or not isinstance(env, BerryWorldEnvironment):
            return

        pos_comp = cast(PositionComponent, pos_comp)
        perc_comp = cast(PerceptionComponent, perc_comp)

        perc_comp.visible_entities.clear()

        # Perceive berries
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
        # Perceive crystals
        for crystal_pos in env.crystal_locations:
            dist = env.distance(pos_comp.position, crystal_pos)
            if dist <= perc_comp.vision_range:
                crystal_id = f"crystal_{crystal_pos[0]}_{crystal_pos[1]}"
                perc_comp.visible_entities[crystal_id] = {
                    "type": "crystal",
                    "position": crystal_pos,
                    "distance": dist,
                }


class BerryActionGenerator(ActionGeneratorInterface):
    """Generates move and eat actions for agents."""

    def generate(
        self, sim_state: AbstractSimulationState, entity_id: str, tick: int
    ) -> List[ActionPlanComponent]:
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


class HeuristicDecisionSelector(DecisionSelectorInterface):
    """Group A: Simple heuristic policy with direct environment access."""

    def __init__(self, simulation_state: Any, config: Any):
        pass

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions or not isinstance(sim_state, SimulationState):
            return None

        eat_actions = [
            a
            for a in possible_actions
            if a.action_type and isinstance(a.action_type, EatBerryAction)
        ]
        if eat_actions:
            return eat_actions[0]

        move_actions = [
            a
            for a in possible_actions
            if a.action_type and isinstance(a.action_type, MoveAction)
        ]
        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        env = sim_state.environment
        if pos_comp and isinstance(env, BerryWorldEnvironment) and move_actions:
            pos_comp = cast(PositionComponent, pos_comp)
            closest_berry_pos = None
            min_dist = float("inf")

            for berry_pos in env.berry_locations.keys():
                dist = env.distance(pos_comp.position, berry_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_berry_pos = berry_pos

            if closest_berry_pos:
                best_move = min(
                    move_actions,
                    key=lambda m: env.distance(
                        m.params["target_pos"], closest_berry_pos
                    ),
                )
                return best_move

        return random.choice(move_actions) if move_actions else None


class ExplorationHeuristicDecisionSelector(HeuristicDecisionSelector):
    """Group D: Heuristic policy with added exploration."""

    def __init__(self, simulation_state: Any, config: Any):
        super().__init__(simulation_state, config)
        self.epsilon = config.learning.q_learning.get("initial_epsilon", 0.1)

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(possible_actions)

        return super().select(sim_state, entity_id, possible_actions)


class QLearningDecisionSelector(DecisionSelectorInterface):
    """Groups B & C: A decision selector that uses the agent's Q-learning network."""

    def __init__(self, simulation_state: Any, config: Any):
        self.simulation_state = simulation_state
        self.config = config
        self.state_encoder = BerryStateEncoder()

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions or not isinstance(sim_state, SimulationState):
            return None

        q_comp = sim_state.get_component(entity_id, QLearningComponent)
        if not q_comp:
            return random.choice(possible_actions)
        q_comp = cast(QLearningComponent, q_comp)

        # Epsilon-greedy exploration
        epsilon = self.config.learning.q_learning.get("initial_epsilon", 0.1)
        if random.random() < epsilon:
            return random.choice(possible_actions)

        # Exploitation
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


class BerryStateEncoder(StateEncoderInterface):
    """Encodes the simulation state into a feature vector for the Q-Learning model."""

    def encode_state(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        config: Any,
        target_entity_id: Optional[str] = None,
    ) -> np.ndarray:
        """Creates a feature vector including vitals, perception, and internal state."""
        if not isinstance(sim_state, SimulationState):
            return np.zeros(14, dtype=np.float32)

        pos_comp = sim_state.get_component(entity_id, PositionComponent)
        health_comp = sim_state.get_component(entity_id, HealthComponent)
        perc_comp = sim_state.get_component(entity_id, PerceptionComponent)
        boost_comp = sim_state.get_component(entity_id, MetabolicBoostComponent)

        env_params = config.environment.params
        width = env_params.width
        height = env_params.height

        # 1. Agent's own state vector
        agent_x = 0.5
        agent_y = 0.5
        pos_comp_cast: Optional[PositionComponent] = None
        if pos_comp:
            pos_comp_cast = cast(PositionComponent, pos_comp)
            agent_x = pos_comp_cast.x / width
            agent_y = pos_comp_cast.y / height

        if health_comp:
            health_comp = cast(HealthComponent, health_comp)
            health = health_comp.current_health / health_comp.initial_health
        else:
            health = 0.5

        if boost_comp:
            boost_comp = cast(MetabolicBoostComponent, boost_comp)
            is_boosted = 1.0 if boost_comp.active else 0.0
        else:
            is_boosted = 0.0

        agent_state_vector = [agent_x, agent_y, health, is_boosted]

        # 2. Agent's perception vector
        visible_items: Dict[
            str, Any
        ] = {}  # Explicit type annotation to fix assignment error
        if perc_comp:
            perc_comp = cast(PerceptionComponent, perc_comp)
            visible_items = perc_comp.visible_entities

        nearest: Dict[str, Optional[Dict[str, Any]]] = {
            "red": None,
            "blue": None,
            "yellow": None,
            "orange": None,
            "crystal": None,
        }

        for item_data in visible_items.values():
            item_type = item_data.get("berry_type") or item_data.get("type")
            if item_type in nearest:
                current_nearest = nearest[item_type]
                if (
                    current_nearest is None
                    or item_data["distance"] < current_nearest["distance"]
                ):
                    nearest[item_type] = item_data

        perception_vector = []
        vision_range = config.agent.vision_range
        for item_type in ["red", "blue", "yellow", "orange", "crystal"]:
            item_data = nearest[item_type]
            if item_data and pos_comp_cast:
                dist = item_data["distance"] / vision_range
                dx = item_data["position"][0] - pos_comp_cast.x
                dy = item_data["position"][1] - pos_comp_cast.y
                angle = math.atan2(dy, dx) / math.pi
                perception_vector.extend([dist, angle])
            else:
                perception_vector.extend([1.0, 0.0])  # Default if not seen

        return np.array(agent_state_vector + perception_vector, dtype=np.float32)

    def encode_internal_state(
        self, components: Dict[Type[Component], Component], config: Any
    ) -> np.ndarray:
        # This simulation doesn't use complex internal state for decisions
        return np.array([0.0], dtype=np.float32)


class BerryRewardCalculator(RewardCalculatorInterface):
    def calculate_final_reward(
        self,
        base_reward: float,
        action_type: Any,
        action_intent: str,
        outcome_details: dict[str, Any],
        entity_components: dict[type[Component], Component],
    ) -> tuple[float, dict[str, Any]]:
        return base_reward, {"base_reward": base_reward}


class BerryComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data for this simulation."""

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        class_name = component_type.split(".")[-1]
        component_map = {
            "PositionComponent": PositionComponent,
            "HealthComponent": HealthComponent,
            "BerryComponent": BerryComponent,
            "WaterComponent": WaterComponent,
            "RockComponent": RockComponent,
            "PurifierCrystalComponent": PurifierCrystalComponent,
            "MetabolicBoostComponent": MetabolicBoostComponent,
            "TimeBudgetComponent": TimeBudgetComponent,
            "QLearningComponent": QLearningComponent,
            "PerceptionComponent": PerceptionComponent,
        }

        if class_name in component_map:
            if class_name == "QLearningComponent":
                return QLearningComponent(
                    state_feature_dim=14,  # 4 (agent) + 5*2 (perception) = 14
                    internal_state_dim=1,
                    action_feature_dim=5,  # move, red, blue, yellow, orange
                    q_learning_alpha=0.1,
                    device=torch.device("cpu"),
                )
            return component_map[class_name](**data)

        raise TypeError(f"Unknown component type for factory: {component_type}")


class BerryVitalityMetricsProvider(VitalityMetricsProviderInterface):
    def get_normalized_vitality_metrics(
        self, entity_id, components, config
    ) -> Dict[str, float]:
        health_comp = components.get(HealthComponent)
        if health_comp:
            health_comp = cast(HealthComponent, health_comp)
            return {
                "health_norm": health_comp.current_health / health_comp.initial_health
            }
        return {"health_norm": 0.5}


class BerryStateNodeEncoder(StateNodeEncoderInterface):
    def __init__(self, simulation_state: SimulationState):
        self.simulation_state = simulation_state

    def encode_state_for_causal_graph(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        current_tick: int,
        config: Any,
    ) -> Tuple[Any, ...]:
        pos_comp = components.get(PositionComponent)
        env = self.simulation_state.environment
        if not pos_comp or not isinstance(env, BerryWorldEnvironment):
            return ("STATE", "unknown_context", "unknown_health")

        pos_comp = cast(PositionComponent, pos_comp)

        context = env.get_environmental_context(pos_comp.position)
        health_comp = components.get(HealthComponent)
        health_status = "healthy"
        if health_comp:
            health_comp = cast(HealthComponent, health_comp)
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
            f"near_crystal_{context['near_crystal']}",
        )
