# src/core/simulation/engine.py

import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Type, cast

import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf  # type: ignore

from src.agents.actions.action_interface import ActionInterface
from src.agents.actions.action_registry import action_registry
from src.agents.actions.base_action import Intent
from src.cognition.ai_models.openai_client import get_client as get_llm_client
from src.cognition.reflection.belief_system import BeliefSystem
from src.cognition.scaffolding import CognitiveScaffold
from src.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    CombatComponent,
    Component,
    EmotionComponent,
    EnvironmentObservationComponent,
    GoalComponent,
    HealthComponent,
    IdentityComponent,
    InventoryComponent,
    MemoryComponent,
    PositionComponent,
    QLearningComponent,
    ResourceComponent,
    SocialMemoryComponent,
    TimeBudgetComponent,
)
from src.core.ecs.event_bus import EventBus
from src.core.ecs.system import SystemManager
from src.core.simulation.scenarios import ScenarioLoader
from src.core.simulation.simulation_state import SimulationState
from src.data.async_database_manager import AsyncDatabaseManager
from src.data.async_runner import async_runner
from src.environment.interface import Grid2DEnvironment
from src.systems.action_system import ActionSystem
from src.systems.affect_system import AffectSystem
from src.systems.causal_graph_system import CausalGraphSystem
from src.systems.combat_system import CombatSystem
from src.systems.component_validator import ComponentValidator
from src.systems.decay_system import DecaySystem
from src.systems.failed_states_system import FailedStatesSystem
from src.systems.goal_system import GoalSystem
from src.systems.identity_system import IdentitySystem
from src.systems.logging_system import LoggingSystem
from src.systems.meta_learning_system import MetaLearningSystem
from src.systems.metrics_system import MetricsSystem
from src.systems.movement_system import MovementSystem
from src.systems.nest_system import NestSystem
from src.systems.q_learning_system import QLearningSystem
from src.systems.reflection_action_system import ReflectionActionSystem
from src.systems.reflection_system import ReflectionSystem
from src.systems.render_system import RenderSystem
from src.systems.resource_system import ResourceSystem
from src.systems.social_interaction_system import SocialInteractionSystem
from src.utils.state_encoding import encode_state

matplotlib.use("Agg")


class SimulationManager:
    """
    Manages the state and execution flow of the multi-entity simulation using an ECS pattern.
    """

    def __init__(
        self,
        config: DictConfig,
        scenario_path: Optional[str] = None,
        task_id: str = "local_run",
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.config: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))

        if run_id:
            self.simulation_id = run_id
        else:
            seed = self.config.get("simulation", {}).get("random_seed")
            timestamp = int(time.time() * 1000000)
            if seed is not None:
                task_suffix = task_id[-8:] if len(task_id) > 8 else task_id
                self.simulation_id = f"sim_{seed:08d}_{task_suffix}_{timestamp % 100000}"
            else:
                self.simulation_id = f"sim_{uuid.uuid4().hex[:16]}"

        self.device = torch.device("cpu")
        print("--- Forcing CPU to avoid MPS fork issue ---")

        seed = self.config.get("simulation", {}).get("random_seed")
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            self.main_rng = np.random.default_rng(seed)
            self.shuffle_rng = np.random.default_rng(seed + 3)
            print(f"--- Simulation running with MASTER SEED: {seed} ---")
        else:
            self.main_rng = np.random.default_rng()
            self.shuffle_rng = np.random.default_rng()

        self.llm_client = get_llm_client()
        self.db_logger = AsyncDatabaseManager()
        scenario_name_str = os.path.basename(scenario_path).replace(".json", "") if scenario_path else "unknown"

        create_run_coro = self.db_logger.create_simulation_run(
            run_id=self.simulation_id,
            experiment_id=experiment_id,  # type: ignore[arg-type]
            scenario_name=scenario_name_str,
            config=self.config,
            task_id=task_id,
        )
        async_runner.run_async(create_run_coro)
        print(f"--- Simulation run {self.simulation_id} created in database. ---")

        self.component_validator = ComponentValidator(self.config)
        self.simulation_state = SimulationState(self.config, self.device)

        grid_width, grid_height = self.config.get("environment", {}).get("grid_world_size", (10, 10))
        self.simulation_state.environment = Grid2DEnvironment(grid_width, grid_height)

        self.simulation_state.simulation_id = self.simulation_id
        self.simulation_state.main_rng = self.main_rng

        self.event_bus = EventBus(self.config)

        self.simulation_state.event_bus = self.event_bus
        self.simulation_state.db_logger = self.db_logger

        self.simulation_state.llm_client = self.llm_client
        # Instantiate the new CognitiveScaffold
        self.cognitive_scaffold = CognitiveScaffold(
            db_logger=self.db_logger,
            simulation_id=self.simulation_id,
            config=self.config,
        )
        self.simulation_state.cognitive_scaffold = self.cognitive_scaffold
        # Pass it to the SystemManager
        self.system_manager = SystemManager(self.simulation_state, self.config, self.cognitive_scaffold)
        self.simulation_state.system_manager = self.system_manager

        self.image_frames_dir = os.path.join(self.config["simulation"]["log_directory"], "frames")
        os.makedirs(self.image_frames_dir, exist_ok=True)
        self._initialize_ecs_systems()

        if scenario_path:
            loader = ScenarioLoader(self.simulation_state, self.config)
            loader.load_and_setup_simulation(scenario_path)
        else:
            print("Warning: No scenario file provided. Initializing empty simulation.")

    def _initialize_ecs_systems(self):
        """Registers all systems with the SystemManager."""
        system_classes = [
            DecaySystem,
            MovementSystem,
            ResourceSystem,
            CombatSystem,
            SocialInteractionSystem,
            AffectSystem,
            ReflectionSystem,
            BeliefSystem,
            IdentitySystem,
            GoalSystem,
            QLearningSystem,
            FailedStatesSystem,
            CausalGraphSystem,
            ActionSystem,
            ReflectionActionSystem,
            RenderSystem,
            MetricsSystem,
            MetaLearningSystem,
            NestSystem,
            LoggingSystem,
        ]
        for system_class in system_classes:
            self.system_manager.register_system(system_class)
        print("All ECS Systems registered.")

    def run(self):
        """Executes the main ECS simulation loop."""
        num_steps = self.config["simulation"]["steps"]
        print(f"\nStarting multi-entity simulation for {num_steps} steps...")

        for step in range(num_steps):
            if step % 10 == 0:
                self._validate_all_entities(step)
            print(f"\n--- Simulation Step {step + 1}/{num_steps} ---")

            active_entities_this_step = [
                entity_id
                for entity_id, components in self.simulation_state.entities.items()
                if components.get(TimeBudgetComponent) and components[TimeBudgetComponent].is_active
            ]

            if not active_entities_this_step:
                print("All entities are inactive. Ending simulation.")
                break

            self.shuffle_rng.shuffle(active_entities_this_step)

            for entity_id in active_entities_this_step:
                time_comp = self.simulation_state.get_component(entity_id, TimeBudgetComponent)

                if not time_comp or not time_comp.is_active:
                    continue

                print(f"    Processing turn for {entity_id}. Current Time Budget: {time_comp.current_time_budget:.1f}")
                self._simulate_entity_decision(entity_id, current_tick=step)

                action_plan_comp = self.simulation_state.get_component(entity_id, ActionPlanComponent)
                if not action_plan_comp or not action_plan_comp.action_type:
                    print(f" {entity_id} has no possible actions and passes the turn.")
                    continue

                self.event_bus.publish(
                    "action_chosen",
                    {
                        "entity_id": entity_id,
                        "action_plan_component": action_plan_comp,
                        "current_tick": step,
                    },
                )

            self.system_manager.update_all(current_tick=step)

            if all(
                not comps.get(TimeBudgetComponent) or not comps[TimeBudgetComponent].is_active
                for comps in self.simulation_state.entities.values()
            ):
                print("All entities are inactive. Ending simulation.")
                break

        print("\nMulti-entity simulation loop finished.")
        self._post_run_summary()
        print("\nFinalizing rendering...")
        render_system = self.system_manager.get_system(RenderSystem)
        if render_system:
            render_system._create_simulation_gif()
        else:
            print("RenderSystem not found, cannot create GIF.")
        self._save_results()

    def _validate_all_entities(self, current_step: int):
        """Validate all entities periodically."""
        validation_failures = 0
        for entity_id, components in list(self.simulation_state.entities.items()):
            if not self.component_validator.validate_entity(entity_id, components):
                validation_failures += 1

        if validation_failures > 0:
            print(
                f"Step {current_step}: Validation found issues in {validation_failures} entities (fixes may have been applied)."
            )

    def _simulate_entity_decision(self, entity_id: str, current_tick: int):
        """
        Simulates the agent's decision-making process by generating and selecting an action.
        """
        essential_components: List[Type[Component]] = [
            PositionComponent,
            TimeBudgetComponent,
            HealthComponent,
            InventoryComponent,
            QLearningComponent,
            EnvironmentObservationComponent,
            CombatComponent,
        ]
        # FIX: Ignore mypy error for this line as it can be a false positive with ABCMeta
        if not all(self.simulation_state.get_component(entity_id, comp) for comp in essential_components):  # type: ignore[arg-type]
            print(f"   Entity {entity_id} missing ESSENTIAL components for decision. Skipping.")
            return

        # Generate all possible action plans using the registry
        possible_action_plans = self._generate_possible_action_plans_for_entity(entity_id, current_tick=current_tick)
        if not possible_action_plans:
            return

        # Select the best action plan using the UtilityNetwork
        chosen_action_plan_comp = self._select_action_plan_for_entity(entity_id, possible_action_plans)

        if chosen_action_plan_comp:
            # Update the entity's ActionPlanComponent with the chosen action
            self.simulation_state.add_component(entity_id, chosen_action_plan_comp)

    def _generate_possible_action_plans_for_entity(
        self, entity_id: str, current_tick: int
    ) -> List[ActionPlanComponent]:
        """
        Generates a list of all executable ActionPlanComponent objects for a given entity
        by iterating through the global action registry.
        """
        all_possible_plans: List[ActionPlanComponent] = []
        all_action_classes = action_registry.get_all_actions()

        # Check entity has required components
        essential_components: List[Type[Component]] = [
            PositionComponent,
            TimeBudgetComponent,
            HealthComponent,
            InventoryComponent,
        ]

        missing_components = []
        for comp_type in essential_components:
            # FIX: Ignore mypy error for this line
            if not self.simulation_state.get_component(entity_id, comp_type):  # type: ignore[arg-type]
                missing_components.append(comp_type.__name__)

        if missing_components:
            print(f"DEBUG: Entity {entity_id} missing components: {missing_components}")
            return []

        for action_class in all_action_classes:
            try:
                action_instance = action_class()
                possible_params_list = action_instance.generate_possible_params(
                    entity_id, self.simulation_state, current_tick
                )  # type: ignore[arg-type]

                for params in possible_params_list:
                    time_cost = action_instance.get_base_cost(self.simulation_state)  # type: ignore[arg-type]
                    plan = ActionPlanComponent(
                        action_type=action_instance,
                        intent=params.get("intent", Intent.SOLITARY),
                        params=params,
                        time_cost=time_cost,
                        chosen_action_id=entity_id,
                    )
                    all_possible_plans.append(plan)

            except Exception as e:
                print(f"DEBUG: Error generating params for action {action_class}: {e}")
                import traceback

                traceback.print_exc()

        return all_possible_plans

    def _select_action_plan_for_entity(
        self, entity_id: str, possible_action_plans: List[ActionPlanComponent]
    ) -> Optional[ActionPlanComponent]:
        """
        Selects an action plan probabilistically using the Q-Learning utility network.
        """
        env_obs_comp = self.simulation_state.get_component(entity_id, EnvironmentObservationComponent)
        if not isinstance(env_obs_comp, EnvironmentObservationComponent):
            return random.choice(possible_action_plans) if possible_action_plans else None

        q_comp = self.simulation_state.get_component(entity_id, QLearningComponent)
        if not isinstance(q_comp, QLearningComponent):
            return random.choice(possible_action_plans) if possible_action_plans else None

        all_action_utilities = []
        with torch.no_grad():
            for action_plan in possible_action_plans:
                # FIX: Add a guard clause for a null action_type
                if not action_plan.action_type:
                    continue

                target_id = action_plan.params.get("target_agent_id")
                state_features = encode_state(
                    self.simulation_state,
                    entity_id,
                    self.config,
                    target_entity_id=target_id,
                )
                state_features_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)

                # FIX: Cast each component to its specific type for the function call.
                id_comp = cast(
                    IdentityComponent,
                    self.simulation_state.get_component(entity_id, IdentityComponent),
                )
                affect_comp = cast(
                    AffectComponent,
                    self.simulation_state.get_component(entity_id, AffectComponent),
                )
                goal_comp = cast(
                    GoalComponent,
                    self.simulation_state.get_component(entity_id, GoalComponent),
                )
                emotion_comp = cast(
                    EmotionComponent,
                    self.simulation_state.get_component(entity_id, EmotionComponent),
                )

                internal_features = self.simulation_state.get_internal_state_features_for_entity(
                    id_comp, affect_comp, goal_comp, emotion_comp
                )
                internal_features_tensor = (
                    torch.tensor(internal_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                )

                # Ensure action_type is an instance of ActionInterface before calling its method
                if isinstance(action_plan.action_type, ActionInterface):
                    action_features = action_plan.action_type.get_feature_vector(
                        entity_id, self.simulation_state, action_plan.params
                    )
                    action_features_tensor = (
                        torch.tensor(action_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                    )

                    utility = q_comp.utility_network(
                        state_features_tensor,
                        internal_features_tensor,
                        action_features_tensor,
                    ).item()
                    all_action_utilities.append((action_plan, utility))

        if not all_action_utilities:
            return None

        if random.random() < q_comp.current_epsilon:
            return random.choice(possible_action_plans)
        else:
            return max(all_action_utilities, key=lambda x: x[1])[0]

    def _post_run_summary(self):
        print("\n--- Initiating final post-run summary for ALL entities ---")
        for entity_id, comps in self.simulation_state.entities.items():
            if comps.get(TimeBudgetComponent):
                print(
                    f"\n  Summarizing for Entity {entity_id} (Status: {'Active' if comps[TimeBudgetComponent].is_active else 'Inactive'})..."
                )
                self.system_manager.get_system(ReflectionSystem).update_for_entity(
                    entity_id,
                    current_tick=max(0, self.config["simulation"]["steps"] - 1),
                    is_final_reflection=True,
                )
        print("\nFinal post-run summary complete.")

    def _save_results(self):
        self.db_logger.close()
        output_file_path = os.path.join(self.config["simulation"]["log_directory"], "final_output_multi_entity.txt")
        print(f"\nSaving final results to {output_file_path}...")

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("--- Multi-Entity Simulation Summary ---\n")
            f.write(f"Simulation ID: {self.simulation_id}\n")
            db_url = os.getenv("DATABASE_URL", "Default DB (SQLite)")
            f.write(f"Data saved to central database: {db_url}\n")
            f.write(f"Total Steps Run: {self.config['simulation']['steps']}\n")

            is_any_agent_active = any(
                c.get(TimeBudgetComponent) and c[TimeBudgetComponent].is_active
                for c in self.simulation_state.entities.values()
            )
            f.write(
                f"Simulation Ended: {'Max Steps Reached' if is_any_agent_active else 'All Entities Ceased Function'}\n"
            )

            f.write("\n\n--- Agent Final States ---\n")
            agent_ids = sorted(
                [eid for eid, comps in self.simulation_state.entities.items() if TimeBudgetComponent in comps]
            )
            for entity_id in agent_ids:
                self._write_agent_summary(f, entity_id, self.simulation_state.entities[entity_id])

            f.write("\n\n--- Resource Final States ---\n")
            resource_ids = sorted(
                [eid for eid, comps in self.simulation_state.entities.items() if ResourceComponent in comps]
            )
            if not resource_ids:
                f.write("No resource nodes were present in the simulation.\n")
            else:
                for res_id in resource_ids:
                    res_comp = self.simulation_state.get_component(res_id, ResourceComponent)
                    pos_comp = self.simulation_state.get_component(res_id, PositionComponent)
                    status = "Depleted" if res_comp.is_depleted else f"Health={res_comp.current_health:.1f}"
                    f.write(f"- Resource {res_id} ({res_comp.type}) at {pos_comp.position}: {status}\n")

        print("Results saved successfully.")

    def _write_agent_summary(self, file_handle, entity_id, components):
        file_handle.write(f"\n{'=' * 20} Entity: {entity_id} {'=' * 20}\n")

        time_comp = components.get(TimeBudgetComponent)
        pos_comp = components.get(PositionComponent)
        health_comp = components.get(HealthComponent)
        inv_comp = components.get(InventoryComponent)
        q_comp = components.get(QLearningComponent)
        goal_comp = components.get(GoalComponent)
        id_comp = components.get(IdentityComponent)
        mem_comp = components.get(MemoryComponent)
        social_mem_comp = components.get(SocialMemoryComponent)

        file_handle.write(f"Status: {'Functioning' if time_comp and time_comp.is_active else 'Non-Functional'}\n")
        if pos_comp:
            file_handle.write(f"Final Position: {pos_comp.position}\n")
        if all([health_comp, time_comp, inv_comp]):
            file_handle.write(
                f"Final Vitals: Health={health_comp.current_health:.2f}, Time={time_comp.current_time_budget:.2f}, Resources={inv_comp.current_resources:.2f}\n"
            )
            file_handle.write(f"Times Inactivated: {time_comp.times_died}\n")

        if goal_comp:
            file_handle.write(f"Final Goal: {goal_comp.current_symbolic_goal or 'N/A'}\n")
        if id_comp:
            file_handle.write(f"Final Traits (Cached): {id_comp.salient_traits_cache or 'N/A'}\n")
        if q_comp:
            file_handle.write(f"Final Epsilon (Exploration Rate): {q_comp.current_epsilon:.4f}\n")

        file_handle.write("\n-- Final Autobiographical Reflection --\n")
        file_handle.write(
            f"{mem_comp.last_llm_reflection_summary if mem_comp and mem_comp.last_llm_reflection_summary else 'No reflection available.'}\n"
        )

        file_handle.write("\n-- Social Schemas (Impressions of Others) --\n")
        if social_mem_comp and social_mem_comp.schemas:
            for other_id, schema in sorted(social_mem_comp.schemas.items()):
                if schema.interaction_count > 0:
                    file_handle.write(
                        f"- Schema for {other_id}: Valence={schema.impression_valence:.2f}, Arousal={schema.impression_arousal:.2f}, Self-Image='{schema.self_schema_in_relation}'\n"
                    )
        else:
            file_handle.write("No social interactions recorded.\n")
