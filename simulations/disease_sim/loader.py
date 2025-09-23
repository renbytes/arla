# simulations/disease_sim/loader.py
"""
Defines the ScenarioLoader for the Disease simulation.
"""

import json
import random
from typing import Any
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface
from .components import (
    DiseaseStateComponent,
    DiseaseStateEnum,
    DiseaseParametersComponent,
    NeighborhoodComponent,
    PositionComponent,
    SocialNetworkComponent,
)
from .environment import DiseaseEnvironment


class DiseaseScenarioLoader(ScenarioLoaderInterface):
    """
    Loads the scenario, creating agents and initializing their disease state.
    """

    def __init__(self, simulation_state: Any, scenario_path: str):
        self.simulation_state = simulation_state
        self.scenario_path = scenario_path

    def load(self) -> None:
        with open(self.scenario_path, "r") as f:
            scenario_data = json.load(f)

        env = self.simulation_state.environment
        if not isinstance(env, DiseaseEnvironment):
            raise TypeError("Environment must be a DiseaseEnvironment instance.")

        num_agents = self.simulation_state.config.agent.foundational.num_agents
        initial_infected = scenario_data.get("initial_infected_count", 1)

        # Create agents at random locations
        locations = random.sample(env.get_valid_positions(), num_agents)

        for i in range(num_agents):
            agent_id = f"neighborhood_{i:03d}"
            position = locations[i]

            self.simulation_state.add_entity(agent_id)
            self.simulation_state.add_component(
                agent_id, PositionComponent(x=position[0], y=position[1])
            )
            # Add all the necessary components for the disease model
            self.simulation_state.add_component(agent_id, DiseaseStateComponent())
            self.simulation_state.add_component(agent_id, NeighborhoodComponent())
            self.simulation_state.add_component(agent_id, SocialNetworkComponent())
            self.simulation_state.add_component(
                agent_id,
                DiseaseParametersComponent(
                    infection_prob_i=self.simulation_state.config.disease.infection_probability_i,
                    infection_prob_e=self.simulation_state.config.disease.infection_probability_e,
                    incubation_period=self.simulation_state.config.disease.incubation_period_mean,
                    infection_period=self.simulation_state.config.disease.infection_period_mean,
                ),
            )
            self.simulation_state.add_component(
                agent_id, TimeBudgetComponent(initial_time_budget=99999)
            )
            env.update_entity_position(agent_id, None, position)

        # Set initial infected agents
        agent_ids = list(self.simulation_state.entities.keys())
        infected_agents = random.sample(agent_ids, initial_infected)
        for agent_id in infected_agents:
            state_comp = self.simulation_state.get_component(
                agent_id, DiseaseStateComponent
            )
            if state_comp:
                state_comp.state = DiseaseStateEnum.INFECTIOUS
