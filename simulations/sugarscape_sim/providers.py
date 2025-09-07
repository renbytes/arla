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
from agent_core.agents.action_cost_provider_interface import ActionCostProviderInterface
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.cognition.scaffolding import CognitiveScaffold
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
        if not hasattr(sim_state, "environment"):
            return

        pos_comp = components.get(PositionComponent)
        metabolism_comp = components.get(MetabolismComponent)
        perc_comp = components.get(PerceptionComponent)
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
        env = cast(SugarscapeEnvironment, env)

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

    def __init__(self, simulation_state: Any = None, config: Any = None):
        self.simulation_state = simulation_state
        self.config = config

    def generate(
        self, sim_state: AbstractSimulationState, entity_id: str, tick: int
    ) -> List[ActionPlanComponent]:
        """Iterates through all registered actions and generates their parameters."""
        all_plans: List[ActionPlanComponent] = []

        is_survival_phase = self.config.simulation.get(
            "learning_phase_survival_only", False
        )

        for action_class in action_registry.get_all_actions():
            action_instance = action_class()

            if is_survival_phase and action_instance.action_id not in [
                "move",
                "harvest",
                "stay",
            ]:
                continue

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
        self.simulation_state = simulation_state
        self.config = config

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        """Selects an action based on a simple survival heuristic."""
        if not possible_actions or not hasattr(sim_state, "environment"):
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

        #  Check if sim_state has the 'environment' attribute before using it.
        if not isinstance(sim_state, SimulationState) or not sim_state.environment:
            return random.choice(possible_actions) if possible_actions else None

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

        move_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "move"
        ]
        return random.choice(move_actions) if move_actions else None


class ExplorationHeuristicDecisionSelector(HeuristicDecisionSelector):
    """Group A (Revised Baseline): Heuristic policy with added exploration."""

    def __init__(self, simulation_state: Any = None, config: Any = None):
        super().__init__(simulation_state, config)
        self.epsilon = config.learning.q_learning.min_epsilon if config else 0.1

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


class RandomDecisionSelector(DecisionSelectorInterface):
    """Group D (Null Baseline): A policy that selects a random valid action."""

    def __init__(self, simulation_state: Any = None, config: Any = None):
        pass

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions:
            return None
        return random.choice(possible_actions)


class QLearningDecisionSelector(DecisionSelectorInterface):
    """A decision selector that uses the agent's Q-learning network."""

    def __init__(self, simulation_state: Any = None, config: Any = None):
        self.simulation_state = simulation_state
        self.config = config
        self.state_encoder = SugarscapeStateEncoder()

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

        epsilon = self.config.learning.q_learning.min_epsilon
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


class LLMHeuristicDecisionSelector(DecisionSelectorInterface):
    """
    Group E (Tactical LLM): Uses an LLM for every tactical decision.
    """

    def __init__(self, simulation_state: Any = None, config: Any = None):
        self.simulation_state = simulation_state
        self.config = config
        self.cognitive_scaffold: Optional[CognitiveScaffold] = (
            simulation_state.cognitive_scaffold if simulation_state else None
        )

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions or not self.cognitive_scaffold:
            return random.choice(possible_actions) if possible_actions else None

        current_tick = (
            sim_state.current_tick if hasattr(sim_state, "current_tick") else 0
        )

        llm_goal = self._get_llm_goal(sim_state, entity_id, current_tick)
        return self._select_action_for_goal(
            llm_goal, possible_actions, sim_state, entity_id
        )

    def _get_llm_goal(
        self, sim_state: AbstractSimulationState, entity_id: str, current_tick: int
    ) -> str:
        prompt = self._build_llm_prompt(sim_state, entity_id)
        if not prompt:
            return "EXPLORE"

        try:
            if not self.cognitive_scaffold:
                return "EXPLORE"
            response = self.cognitive_scaffold.query(
                agent_id=entity_id,
                purpose="tactical_goal_selection",
                prompt=prompt,
                current_tick=current_tick,
            )
            goal = response.strip().upper()
            return "HARVEST" if "HARVEST" in goal else "EXPLORE"
        except Exception as e:
            print(f"Tactical LLM goal selection failed for {entity_id}: {e}")
        return "EXPLORE"

    def _build_llm_prompt(
        self, sim_state: AbstractSimulationState, entity_id: str
    ) -> Optional[str]:
        energy_comp = cast(
            EnergyComponent, sim_state.get_component(entity_id, EnergyComponent)
        )
        perc_comp = cast(
            PerceptionComponent, sim_state.get_component(entity_id, PerceptionComponent)
        )
        if not energy_comp or not perc_comp:
            return None

        status = f"My current energy is {energy_comp.current_energy:.0f} out of {energy_comp.initial_energy:.0f}."
        visible_sugar = [
            v for v in perc_comp.visible_entities.values() if v["type"] == "sugar"
        ]
        perception = (
            f"I see {len(visible_sugar)} sugar patches."
            if visible_sugar
            else "I cannot see any sugar."
        )

        return f"""
        Given my situation, what is my single most important priority?
        Choose ONE: HARVEST or EXPLORE.

        Situation:
        - {status}
        - {perception}

        Priority:
        """

    def _select_action_for_goal(
        self,
        goal: str,
        possible_actions: List[ActionPlanComponent],
        sim_state: AbstractSimulationState,
        entity_id: str,
    ) -> Optional[ActionPlanComponent]:
        """Heuristic to execute a tactical goal."""
        if goal == "HARVEST":
            harvest_actions = [
                a
                for a in possible_actions
                if a.action_type and a.action_type.action_id == "harvest"
            ]
            if harvest_actions:
                return harvest_actions[0]

        move_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "move"
        ]
        if not move_actions:
            #  Check if action_type is not None before accessing action_id.
            return next(
                (
                    a
                    for a in possible_actions
                    if a.action_type and a.action_type.action_id == "stay"
                ),
                None,
            )

        perc_comp = cast(
            PerceptionComponent, sim_state.get_component(entity_id, PerceptionComponent)
        )

        #  Check if sim_state has the 'environment' attribute.
        if not isinstance(sim_state, SimulationState) or not sim_state.environment:
            return random.choice(move_actions)

        env = cast(SugarscapeEnvironment, sim_state.environment)

        if perc_comp and env:
            richest_patch_pos = None
            max_sugar = -1
            for v in perc_comp.visible_entities.values():
                if v["type"] == "sugar" and v["amount"] > max_sugar:
                    max_sugar = v["amount"]
                    richest_patch_pos = v["position"]

            if richest_patch_pos:
                return min(
                    move_actions,
                    key=lambda m: env.distance(
                        m.params["target_pos"], richest_patch_pos
                    ),
                )
        return random.choice(move_actions)


class StrategicLLMDecisionSelector(DecisionSelectorInterface):
    """
    Group F (Strategic LLM): Uses an LLM for periodic strategic planning.
    """

    def __init__(self, simulation_state: Any = None, config: Any = None):
        self.simulation_state = simulation_state
        self.config = config
        self.cognitive_scaffold: Optional[CognitiveScaffold] = (
            simulation_state.cognitive_scaffold if simulation_state else None
        )
        self.strategic_planning_interval = 20
        self.last_plan_tick: Dict[str, int] = {}
        self.current_plan: Dict[str, Dict[str, Any]] = {}

    def select(
        self,
        sim_state: AbstractSimulationState,
        entity_id: str,
        possible_actions: List[ActionPlanComponent],
    ) -> Optional[ActionPlanComponent]:
        if not possible_actions or not self.cognitive_scaffold:
            return random.choice(possible_actions) if possible_actions else None

        current_tick = (
            sim_state.current_tick if hasattr(sim_state, "current_tick") else 0
        )
        last_plan_tick = self.last_plan_tick.get(
            entity_id, -self.strategic_planning_interval
        )

        if current_tick - last_plan_tick >= self.strategic_planning_interval:
            plan = self._get_llm_strategic_plan(sim_state, entity_id, current_tick)
            self.current_plan[entity_id] = plan
            self.last_plan_tick[entity_id] = current_tick

        active_plan = self.current_plan.get(entity_id)
        if not active_plan:
            return random.choice(possible_actions)

        return self._select_action_for_plan(
            active_plan, possible_actions, sim_state, entity_id
        )

    def _get_llm_strategic_plan(
        self, sim_state: AbstractSimulationState, entity_id: str, current_tick: int
    ) -> Dict[str, Any]:
        """Queries an LLM for a long-term strategic plan."""
        prompt = self._build_llm_strategic_prompt(sim_state, entity_id)
        default_plan = {"type": "EXPLORE_RANDOMLY"}
        if not prompt:
            return default_plan

        try:
            if not self.cognitive_scaffold:
                return default_plan
            response = self.cognitive_scaffold.query(
                agent_id=entity_id,
                purpose="long_term_strategy",
                prompt=prompt,
                current_tick=current_tick,
            )
            plan_str = response.strip().upper()
            if "MIGRATE_TO_NW_PEAK" in plan_str:
                return {"type": "MIGRATE_TO_PEAK", "target": "NW"}
            if "MIGRATE_TO_SE_PEAK" in plan_str:
                return {"type": "MIGRATE_TO_PEAK", "target": "SE"}
            if "HARVEST_LOCALLY" in plan_str:
                return {"type": "HARVEST_LOCALLY"}
        except Exception as e:
            print(f"Strategic LLM plan selection failed for {entity_id}: {e}")

        return default_plan

    def _build_llm_strategic_prompt(
        self, sim_state: AbstractSimulationState, entity_id: str
    ) -> Optional[str]:
        """Constructs a prompt for the LLM for strategic planning."""
        energy_comp = cast(
            EnergyComponent, sim_state.get_component(entity_id, EnergyComponent)
        )
        perc_comp = cast(
            PerceptionComponent, sim_state.get_component(entity_id, PerceptionComponent)
        )
        if not energy_comp or not perc_comp:
            return None

        status = f"My energy is {energy_comp.current_energy:.0f}/{energy_comp.initial_energy:.0f}."
        visible_sugar = [
            v for v in perc_comp.visible_entities.values() if v["type"] == "sugar"
        ]
        perception = (
            f"I see {len(visible_sugar)} sugar patches."
            if visible_sugar
            else "I see no sugar."
        )

        return f"""
        You are an agent in a survival simulation. The world is a grid with two
        major resource peaks, one in the northwest (NW) and one in the southeast (SE).
        What is your long-term strategic plan for the next 20-30 steps?

        Choose ONE from: MIGRATE_TO_NW_PEAK, MIGRATE_TO_SE_PEAK, HARVEST_LOCALLY, EXPLORE_RANDOMLY.

        My Situation:
        - {status}
        - {perception}

        My Strategic Plan:
        """

    def _select_action_for_plan(
        self,
        plan: Dict[str, Any],
        possible_actions: List[ActionPlanComponent],
        sim_state: AbstractSimulationState,
        entity_id: str,
    ) -> Optional[ActionPlanComponent]:
        """Heuristic to execute a strategic plan."""
        move_actions = [
            a
            for a in possible_actions
            if a.action_type and a.action_type.action_id == "move"
        ]

        #  Check if sim_state has the 'environment' attribute.
        if not isinstance(sim_state, SimulationState) or not sim_state.environment:
            return random.choice(move_actions) if move_actions else None

        env = cast(SugarscapeEnvironment, sim_state.environment)

        if plan["type"] == "HARVEST_LOCALLY":
            # Same logic as the heuristic selector
            harvest_actions = [
                a
                for a in possible_actions
                if a.action_type and a.action_type.action_id == "harvest"
            ]
            if harvest_actions:
                return harvest_actions[0]
            if not move_actions:
                #  Check if action_type is not None before accessing action_id.
                return next(
                    (
                        a
                        for a in possible_actions
                        if a.action_type and a.action_type.action_id == "stay"
                    ),
                    None,
                )

            perc_comp = cast(
                PerceptionComponent,
                sim_state.get_component(entity_id, PerceptionComponent),
            )
            if perc_comp:
                richest_patch = max(
                    (
                        v
                        for v in perc_comp.visible_entities.values()
                        if v["type"] == "sugar"
                    ),
                    key=lambda p: p["amount"],
                    default=None,
                )
                if richest_patch:
                    return min(
                        move_actions,
                        key=lambda m: env.distance(
                            m.params["target_pos"], richest_patch["position"]
                        ),
                    )
            return random.choice(move_actions)

        elif plan["type"] == "MIGRATE_TO_PEAK":
            if not move_actions:
                #  Check if action_type is not None before accessing action_id.
                return next(
                    (
                        a
                        for a in possible_actions
                        if a.action_type and a.action_type.action_id == "stay"
                    ),
                    None,
                )
            target_corner = (
                (env.width * 0.25, env.height * 0.25)
                if plan["target"] == "NW"
                else (env.width * 0.75, env.height * 0.75)
            )
            return min(
                move_actions,
                key=lambda m: env.distance(m.params["target_pos"], target_corner),
            )

        elif plan["type"] == "EXPLORE_RANDOMLY":
            #  Check if action_type is not None before accessing action_id.
            return (
                random.choice(move_actions)
                if move_actions
                else next(
                    (
                        a
                        for a in possible_actions
                        if a.action_type and a.action_type.action_id == "stay"
                    ),
                    None,
                )
            )

        return random.choice(possible_actions) if possible_actions else None


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
        if not hasattr(sim_state, "environment"):
            return np.zeros(9, dtype=np.float32)

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
            return np.zeros(9, dtype=np.float32)

        pos_comp = cast(PositionComponent, pos_comp)
        energy_comp = cast(EnergyComponent, energy_comp)
        perc_comp = cast(PerceptionComponent, perc_comp)
        metabolism_comp = cast(MetabolismComponent, metabolism_comp)
        env = cast(SugarscapeEnvironment, env)

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

    def __init__(self, simulation_state: Any, config: Any):
        self.simulation_state = simulation_state
        self.config = config

    def _get_distance_to_nearest_sugar(self, pos, env):
        min_dist = float("inf")
        # Inefficient, but suitable for this simulation's scale
        for y in range(env.height):
            for x in range(env.width):
                if env.get_sugar_at((x, y)) > 0:
                    dist = env.distance(pos, (x, y))
                    if dist < min_dist:
                        min_dist = dist
        return min_dist

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

        reward_shaping_bonus = 0.0
        shaping_enabled = self.config.simulation.get("reward_shaping_enabled", False)

        #  Check if action_type is not None before accessing action_id.
        if shaping_enabled and action_type and action_type.action_id == "move":
            #  Check if self.simulation_state has the 'environment' attribute.
            if (
                isinstance(self.simulation_state, SimulationState)
                and self.simulation_state.environment
            ):
                env = self.simulation_state.environment
                old_pos = outcome_details.get("old_pos")
                new_pos = outcome_details.get("target_pos")

                if old_pos and new_pos and isinstance(env, SugarscapeEnvironment):
                    old_dist = self._get_distance_to_nearest_sugar(old_pos, env)
                    new_dist = self._get_distance_to_nearest_sugar(new_pos, env)
                    reward_shaping_bonus = (old_dist - new_dist) * 0.1

        final_reward = base_reward + survival_penalty + reward_shaping_bonus
        return final_reward, {
            "base_reward": base_reward,
            "survival_penalty": survival_penalty,
            "reward_shaping_bonus": reward_shaping_bonus,
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
            env = cast(SugarscapeEnvironment, env)
            sugar_level = env.get_sugar_at(pos_comp.position)
            if sugar_level > env.max_sugar_per_cell / 2:
                local_sugar_status = "abundant"
            elif sugar_level > 0:
                local_sugar_status = "present"

        return ("STATE", f"energy_{energy_status}", f"sugar_{local_sugar_status}")


class SugarscapeActionCostProvider(ActionCostProviderInterface):
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
