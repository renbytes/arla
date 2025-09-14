"""
Defines the ScenarioLoader for the Tragedy of the Commons simulation.

This class reads a scenario JSON file and populates the simulation state
with the initial set of agents (Herders) and resource patches (Grass).
"""

import json
from typing import Any

import numpy as np
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface

from .components import EnergyComponent, PositionComponent, ResourceComponent
from .environment import CommonsEnvironment


class CommonsScenarioLoader(ScenarioLoaderInterface):
    """
    Loads the Tragedy of the Commons scenario, creating agents and grass patches.
    """

    def __init__(
        self, simulation_state: Any, scenario_path: str, rng: np.random.Generator
    ):
        self.simulation_state = simulation_state
        self.scenario_path = scenario_path
        self.rng = rng

    def load(self) -> None:
        """
        Loads the scenario, initializes the environment, and creates entities.
        """
        with open(self.scenario_path, "r") as f:
            scenario_data = json.load(f)

        env = self.simulation_state.environment
        if not isinstance(env, CommonsEnvironment):
            raise TypeError("Environment must be a CommonsEnvironment instance.")

        # Create resource patches (grass) for every cell
        for y in range(env.height):
            for x in range(env.width):
                resource_id = f"grass_{x}_{y}"
                self.simulation_state.add_entity(resource_id)
                self.simulation_state.add_component(
                    resource_id,
                    ResourceComponent(
                        current_resource=self.simulation_state.config.environment.initial_resource_level,
                        max_resource=self.simulation_state.config.environment.max_resource_per_patch,
                        regeneration_rate=self.simulation_state.config.environment.resource_regeneration_rate,
                    ),
                )
                env.resource_grid[(x, y)] = resource_id

        # Create agents (Herders)
        num_agents = scenario_data["num_agents"]
        initial_energy = self.simulation_state.config.agent.initial_energy
        empty_cells = env.get_all_empty_cells()
        start_positions = self.rng.choice(
            empty_cells, size=min(num_agents, len(empty_cells)), replace=False
        )

        for i, pos_tuple in enumerate(start_positions):
            pos = tuple(pos_tuple)
            agent_id = f"herder_{i:03d}"
            self.simulation_state.add_entity(agent_id)
            self.simulation_state.add_component(
                agent_id, PositionComponent(x=pos[0], y=pos[1])
            )
            self.simulation_state.add_component(
                agent_id,
                EnergyComponent(
                    current_energy=initial_energy, initial_energy=initial_energy
                ),
            )
            self.simulation_state.add_component(
                agent_id, TimeBudgetComponent(initial_time_budget=99999)
            )
            env.update_entity_position(agent_id, None, pos)
