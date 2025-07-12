# simulations/soul_sim/simulation/scenario_loader.py

import json
from typing import Any, Callable, Dict, List, Tuple, Type

from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import (
    ActionOutcomeComponent, ActionPlanComponent, AffectComponent, BeliefSystemComponent,
    CompetenceComponent, EmotionComponent, EpisodeComponent, GoalComponent,
    IdentityComponent, MemoryComponent, SocialMemoryComponent, TimeBudgetComponent,
    ValidationComponent, ValueSystemComponent
)
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface
from agent_engine.cognition.identity.domain_identity import IdentityDomain, MultiDomainIdentity
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent
from agent_engine.utils.config_utils import get_config_value
from simulations.soul_sim.components import (
    CombatComponent, EnvironmentObservationComponent, FailedStatesComponent,
    HealthComponent, InventoryComponent, NestComponent, PositionComponent, ResourceComponent
)


class ScenarioLoader(ScenarioLoaderInterface):
    """
    Loads a scenario from a JSON file and sets up the initial state of the simulation
    by creating agents with pre-defined, robust component archetypes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.simulation_state: SimulationState = None  # Injected by SimulationManager
        self.scenario_data: Dict[str, Any] = {}
        self._define_archetypes()

    def load(self) -> None:
        """Loads the scenario file and populates the simulation state."""
        if not self.simulation_state:
            raise RuntimeError("SimulationState must be set before calling load().")

        scenario_path = self.config.get("scenario_path")
        if not scenario_path:
            raise ValueError("Scenario path not found in configuration.")

        print(f"--- Loading scenario from: {scenario_path} ---")
        with open(scenario_path, "r") as f:
            self.scenario_data = json.load(f)

        self._create_resources()
        self._create_agents()
        print(f"--- Scenario '{self.scenario_data.get('name', 'Untitled')}' loaded successfully. ---")

    def _define_archetypes(self):
        """
        Defines agent archetypes using lists of component-creation functions.
        This is more robust than parsing component paths from YAML.
        """
        CORE_COMPONENTS = [
            lambda kwargs: PositionComponent(kwargs["initial_pos"], kwargs["environment"]),
            lambda kwargs: TimeBudgetComponent(**kwargs["TimeBudgetComponent"]),
            lambda kwargs: HealthComponent(**kwargs["HealthComponent"]),
            lambda kwargs: InventoryComponent(**kwargs["InventoryComponent"]),
            lambda kwargs: CombatComponent(**kwargs["CombatComponent"]),
            lambda kwargs: NestComponent(),
            lambda kwargs: QLearningComponent(**kwargs["QLearningComponent"]),
            lambda kwargs: ActionPlanComponent(),
            lambda kwargs: ActionOutcomeComponent(),
            lambda kwargs: EnvironmentObservationComponent(),
            lambda kwargs: FailedStatesComponent(),
            lambda kwargs: CompetenceComponent(),
        ]

        MEMORY_SUITE = [lambda kwargs: MemoryComponent()]
        AFFECT_SUITE = [
            lambda kwargs: EmotionComponent(),
            lambda kwargs: AffectComponent(**kwargs["AffectComponent"]),
            lambda kwargs: ValueSystemComponent(),
            lambda kwargs: SocialMemoryComponent(**kwargs["SocialMemoryComponent"]),
        ]
        GOAL_SUITE = [lambda kwargs: GoalComponent(**kwargs["GoalComponent"])]
        IDENTITY_SUITE = [
            lambda kwargs: IdentityComponent(**kwargs["IdentityComponent"]),
            lambda kwargs: EpisodeComponent(),
            lambda kwargs: BeliefSystemComponent(),
            lambda kwargs: ValidationComponent(),
        ]

        self.ARCHETYPES = {
            "full_agent": CORE_COMPONENTS + MEMORY_SUITE + AFFECT_SUITE + GOAL_SUITE + IDENTITY_SUITE
        }

    def _create_agents(self):
        """Creates agent entities based on the 'groups' defined in the scenario file."""
        agent_groups = self.scenario_data.get("groups", [])
        agent_counter = 0

        if agent_groups:
            valid_positions = self.simulation_state.environment.get_valid_positions()
            for group in agent_groups:
                archetype_name = group.get("type", "full_agent")
                component_factory_funcs = self.ARCHETYPES.get(archetype_name)

                if not component_factory_funcs:
                    print(f"Warning: Archetype '{archetype_name}' not found. Agent group will be skipped.")
                    continue

                for _ in range(group.get("count", 0)):
                    entity_id = f"{archetype_name}_{agent_counter}"
                    initial_pos = tuple(valid_positions[agent_counter % len(valid_positions)])
                    self.simulation_state.add_entity(entity_id)
                    component_kwargs = self._prepare_component_kwargs(initial_pos)

                    for factory_func in component_factory_funcs:
                        component = factory_func(component_kwargs)
                        self.simulation_state.add_component(entity_id, component)

                    self.simulation_state.environment.update_entity_position(entity_id, None, initial_pos)
                    agent_counter += 1

        # FIX: Moved this print statement outside the conditional to ensure it always runs.
        print(f"--- Created {agent_counter} agents. ---")


    def _create_resources(self):
        """Initializes resource entities based on scenario config."""
        resource_config = self.scenario_data.get("resources", {})
        for i, res_data in enumerate(resource_config.get("resource_list", [])):
            res_id = f"resource_{i}"
            self.simulation_state.add_entity(res_id)
            pos = tuple(res_data["pos"])
            self.simulation_state.add_component(
                res_id, PositionComponent(position=pos, environment=self.simulation_state.environment)
            )
            self.simulation_state.add_component(res_id, ResourceComponent(**res_data["params"]))
            self.simulation_state.environment.update_entity_position(res_id, None, pos)

    def _prepare_component_kwargs(self, initial_pos: tuple) -> Dict[str, Any]:
        """Gathers all possible constructor arguments for any component from the config."""
        main_emb_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)

        # Calculate the true dimensions of the feature vectors dynamically.
        # State features from SoulSimStateEncoder
        state_feature_dim = 16

        # Action features from create_standard_feature_vector
        action_feature_dim = len(action_registry.action_ids) + len(Intent) + 1 + 5

        # Internal state features from get_internal_state_features_for_entity
        internal_state_dim = (
            4  # affect/emotion
            + main_emb_dim  # goal embedding
            + (len(IdentityDomain) * main_emb_dim)  # all identity domains
        )
        return {
            "initial_pos": initial_pos,
            "environment": self.simulation_state.environment,
            "TimeBudgetComponent": {
                "initial_time_budget": get_config_value(self.config, "agent.foundational.vitals.initial_time_budget"),
                "lifespan_std_dev_percent": get_config_value(self.config, "agent.foundational.lifespan_std_dev_percent"),
            },
            "HealthComponent": {"initial_health": get_config_value(self.config, "agent.foundational.vitals.initial_health")},
            "InventoryComponent": {"initial_resources": get_config_value(self.config, "agent.foundational.vitals.initial_resources")},
            "CombatComponent": {"attack_power": get_config_value(self.config, "agent.foundational.attributes.initial_attack_power")},
            "AffectComponent": {"affective_buffer_maxlen": get_config_value(self.config, "learning.memory.affective_buffer_maxlen")},
            "GoalComponent": {"embedding_dim": main_emb_dim},
            "IdentityComponent": {"multi_domain_identity": MultiDomainIdentity(embedding_dim=main_emb_dim)},
            "SocialMemoryComponent": {
                "schema_embedding_dim": get_config_value(self.config, "agent.cognitive.embeddings.schema_embedding_dim"),
                "device": self.simulation_state.device,
            },
            "QLearningComponent": {
                # Use the dynamically calculated dimensions instead of placeholders
                "state_feature_dim": state_feature_dim,
                "internal_state_dim": internal_state_dim,
                "action_feature_dim": action_feature_dim,
                "q_learning_alpha": get_config_value(self.config, "learning.q_learning.alpha"),
                "device": self.simulation_state.device,
            },
        }
