# FILE: simulations/emergence_sim/simulation/scenario_loader.py

import json
import random
from typing import Any

from agent_core.core.ecs.component import (
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
from agent_engine.cognition.identity.domain_identity import MultiDomainIdentity
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.systems.components import QLearningComponent

from simulations.emergence_sim.components import (
    ConceptualSpaceComponent,
    DebtLedgerComponent,
    InventoryComponent,
    PositionComponent,
    RitualComponent,
    SocialCreditComponent,
)


class EmergenceScenarioLoader(ScenarioLoaderInterface):
    """Loads the initial state for the emergence simulation from a JSON file."""

    def __init__(self, config: Any, environment: Any, simulation_state: SimulationState):
        self.config = config
        self.environment = environment
        self.simulation_state = simulation_state

    def load(self):
        """Reads the scenario file and populates the simulation state."""
        try:
            # The scenario path is now optional, handle case where it's not in config
            scenario_path = getattr(
                self.config,
                "scenario_path",
                "simulations/emergence_sim/scenarios/default.json",
            )
            with open(scenario_path, "r") as f:
                scenario_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"FATAL: Could not load or parse scenario file: {e}")
            raise

        for agent_def in scenario_data.get("agents", []):
            if agent_def.get("archetype") == "default":
                # Use the correct config path for the number of agents
                for i in range(agent_def.get("count", self.config.agent.foundational.num_agents)):
                    self._create_default_agent(f"agent_{i + 1}")

    def _create_default_agent(self, agent_id: str):
        """Creates a single agent with all necessary components."""
        self.simulation_state.add_entity(agent_id)

        start_pos = (
            random.randint(0, self.environment.width - 1),
            random.randint(0, self.environment.height - 1),
        )
        self.environment.update_entity_position(agent_id, None, start_pos)

        # Add all required components with initial values from config
        self.simulation_state.add_component(
            agent_id,
            TimeBudgetComponent(
                self.config.agent.foundational.vitals.initial_time_budget,
                lifespan_std_dev_percent=self.config.agent.foundational.lifespan_std_dev_percent,
            ),
        )
        self.simulation_state.add_component(agent_id, PositionComponent(start_pos, self.environment))
        self.simulation_state.add_component(
            agent_id,
            InventoryComponent(self.config.agent.foundational.vitals.initial_resources),
        )

        # Add cognitive and custom components
        self.simulation_state.add_component(agent_id, MemoryComponent())
        self.simulation_state.add_component(agent_id, IdentityComponent(MultiDomainIdentity()))
        # Note: embedding_dim is not in emergence_config, using a reasonable default
        self.simulation_state.add_component(agent_id, GoalComponent(embedding_dim=128))
        self.simulation_state.add_component(agent_id, EmotionComponent())
        # Note: affective_buffer_maxlen is not in emergence_config, using a reasonable default
        self.simulation_state.add_component(agent_id, AffectComponent(affective_buffer_maxlen=200))
        self.simulation_state.add_component(agent_id, CompetenceComponent())
        self.simulation_state.add_component(agent_id, EpisodeComponent())
        self.simulation_state.add_component(agent_id, BeliefSystemComponent())
        self.simulation_state.add_component(agent_id, ValidationComponent())
        self.simulation_state.add_component(agent_id, ValueSystemComponent())
        # Note: schema_embedding_dim is not in emergence_config, using a reasonable default
        self.simulation_state.add_component(agent_id, SocialMemoryComponent(schema_embedding_dim=128, device="cpu"))

        # Add emergence-specific components
        self.simulation_state.add_component(
            agent_id,
            ConceptualSpaceComponent(quality_dimensions={"color": 3, "shape": 3}),
        )
        self.simulation_state.add_component(agent_id, RitualComponent())
        self.simulation_state.add_component(agent_id, DebtLedgerComponent())
        self.simulation_state.add_component(agent_id, SocialCreditComponent())

        # Add Q-learning component using dimensions from the config
        q_config = self.config.learning.q_learning
        self.simulation_state.add_component(
            agent_id,
            QLearningComponent(
                state_feature_dim=q_config.state_feature_dim,
                internal_state_dim=q_config.internal_state_dim,
                action_feature_dim=q_config.action_feature_dim,
                # CORRECTED: The config uses 'alpha', not 'learning_rate'
                q_learning_alpha=q_config.alpha,
                device="cpu",
            ),
        )

        print(f"Created agent '{agent_id}' at position {start_pos}")
