# simulations/soul_sim/simulation/scenario_loader.py

import json
from typing import Any, Dict

from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent
from agent_core.core.ecs.component import (
    ActionOutcomeComponent,
    ActionPlanComponent,
    AffectComponent,
    BeliefSystemComponent,
    CompetenceComponent,
    EmotionComponent,
    EpisodeComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
    SocialMemoryComponent,
    TimeBudgetComponent,
    ValidationComponent,
    ValueSystemComponent,
)
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface
from agent_engine.cognition.identity.domain_identity import (
    IdentityDomain,
    MultiDomainIdentity,
)
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent
from agent_engine.utils.config_utils import get_config_value

from simulations.soul_sim.components import (
    CombatComponent,
    EnvironmentObservationComponent,
    FailedStatesComponent,
    HealthComponent,
    InventoryComponent,
    NestComponent,
    PositionComponent,
    ResourceComponent,
)
from simulations.soul_sim.environment.resources import init_resources


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
        core_components = [
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

        memory_suite = [lambda kwargs: MemoryComponent()]
        affect_suite = [
            lambda kwargs: EmotionComponent(),
            lambda kwargs: AffectComponent(**kwargs["AffectComponent"]),
            lambda kwargs: ValueSystemComponent(),
            lambda kwargs: SocialMemoryComponent(**kwargs["SocialMemoryComponent"]),
        ]
        goal_suite = [lambda kwargs: GoalComponent(**kwargs["GoalComponent"])]
        identity_suite = [
            lambda kwargs: IdentityComponent(**kwargs["IdentityComponent"]),
            lambda kwargs: EpisodeComponent(),
            lambda kwargs: BeliefSystemComponent(),
            lambda kwargs: ValidationComponent(),
        ]

        self.archetypes = {"full_agent": core_components + memory_suite + affect_suite + goal_suite + identity_suite}

    def _create_agents(self):
        """Creates agent entities based on the 'groups' defined in the scenario file."""
        agent_groups = self.scenario_data.get("groups", [])
        agent_counter = 0

        if agent_groups:
            valid_positions = self.simulation_state.environment.get_valid_positions()
            for group in agent_groups:
                archetype_name = group.get("type", "full_agent")
                component_factory_funcs = self.archetypes.get(archetype_name)

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
        """Initializes resource entities by calling the random generator."""
        # Get resource counts from the main YAML config
        num_single = get_config_value(self.config, "environment.num_single_resources", 10)
        num_double = get_config_value(self.config, "environment.num_double_resources", 5)
        num_triple = get_config_value(self.config, "environment.num_triple_resources", 2)
        seed = get_config_value(self.config, "simulation.random_seed", None)

        # Generate the dictionary of resources with random positions
        resources_to_create = init_resources(
            environment=self.simulation_state.environment,
            num_single=num_single,
            num_double=num_double,
            num_triple=num_triple,
            seed=seed,
        )

        # Create entities from the generated dictionary
        for res_id, res_data in resources_to_create.items():
            self.simulation_state.add_entity(res_id)
            pos = tuple(res_data["pos"])

            self.simulation_state.add_component(
                res_id,
                PositionComponent(position=pos, environment=self.simulation_state.environment),
            )

            # FIX: Create a clean dictionary for the component's constructor
            # This prevents passing unexpected arguments like 'id' or 'pos'.
            component_params = {
                "resource_type": res_data["type"],
                "initial_health": res_data["initial_health"],
                "min_agents": res_data["min_agents_needed"],
                "max_agents": res_data["max_agents_allowed"],
                "mining_rate": res_data["mining_rate"],
                "reward_per_mine": res_data["reward_per_mine_action"],
                "resource_yield": res_data["resource_yield"],
                "respawn_time": res_data["resource_respawn_time"],
            }

            self.simulation_state.add_component(res_id, ResourceComponent(**component_params))
            self.simulation_state.environment.update_entity_position(res_id, None, pos)

        print(f"--- Created {len(resources_to_create)} resources randomly. ---")

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
                "lifespan_std_dev_percent": get_config_value(
                    self.config, "agent.foundational.lifespan_std_dev_percent"
                ),
            },
            "HealthComponent": {
                "initial_health": get_config_value(self.config, "agent.foundational.vitals.initial_health")
            },
            "InventoryComponent": {
                "initial_resources": get_config_value(self.config, "agent.foundational.vitals.initial_resources")
            },
            "CombatComponent": {
                "attack_power": get_config_value(self.config, "agent.foundational.attributes.initial_attack_power")
            },
            "AffectComponent": {
                "affective_buffer_maxlen": get_config_value(self.config, "learning.memory.affective_buffer_maxlen")
            },
            "GoalComponent": {"embedding_dim": main_emb_dim},
            "IdentityComponent": {"multi_domain_identity": MultiDomainIdentity(embedding_dim=main_emb_dim)},
            "SocialMemoryComponent": {
                "schema_embedding_dim": get_config_value(
                    self.config, "agent.cognitive.embeddings.schema_embedding_dim"
                ),
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
