# FILE: simulations/sugarscape_sim/loader.py
"""
Defines the ScenarioLoader for the Sugarscape simulation.

This class is responsible for reading a scenario JSON file, parsing the
agent archetypes and distribution, and populating the simulation state
with the initial set of agents and their components. This version is
updated to use a seeded random number generator for deterministic agent placement.
"""

import json
from typing import Any

import numpy as np
from agent_core.core.ecs.component import (
    PerceptionComponent,
    TimeBudgetComponent,
)
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface

from .components import (
    CommunicationComponent,
    EnergyComponent,
    MetabolismComponent,
    PositionComponent,
)
from .environment import SugarscapeEnvironment


class SugarscapeScenarioLoader(ScenarioLoaderInterface):
    """
    Loads the Sugarscape scenario, creating agents based on defined archetypes
    and placing them in a reproducible, uniform grid.
    """

    def __init__(
        self, simulation_state: Any, scenario_path: str, rng: np.random.Generator
    ):
        self.simulation_state = simulation_state
        self.scenario_path = scenario_path
        self.rng = rng

    def load(self) -> None:
        """
        Loads the scenario, initializes agents, and places them in a
        deterministic, uniform grid to reduce environmental variance.
        """
        with open(self.scenario_path, "r") as f:
            scenario_data = json.load(f)

        env = self.simulation_state.environment
        if not isinstance(env, SugarscapeEnvironment):
            raise TypeError("Environment must be a SugarscapeEnvironment instance.")

        archetypes = {arch["name"]: arch for arch in scenario_data["agent_archetypes"]}
        agent_distribution = scenario_data["agent_distribution"]
        total_agents = sum(agent_distribution.values())

        # This creates a deterministic, grid-based spawn pattern.
        spawn_locations = []
        if total_agents > 0:
            grid_side = int(np.ceil(np.sqrt(total_agents)))
            spacing_x = env.width // (grid_side + 1)
            spacing_y = env.height // (grid_side + 1)

            for i in range(total_agents):
                row = (i // grid_side) + 1
                col = (i % grid_side) + 1
                x, y = col * spacing_x, row * spacing_y
                if env.is_valid_position((x, y)) and not env.get_entities_at_position(
                    (x, y)
                ):
                    spawn_locations.append((x, y))

        # Shuffle the deterministic locations to randomly assign archetypes
        self.rng.shuffle(spawn_locations)
        spawn_iterator = iter(spawn_locations)

        agent_counter = 0
        for archetype_name, count in agent_distribution.items():
            if count == 0:
                continue

            archetype = archetypes[archetype_name]
            for _ in range(count):
                agent_id = f"{archetype_name}_{agent_counter:03d}"
                agent_counter += 1

                self.simulation_state.add_entity(agent_id)

                try:
                    start_pos = next(spawn_iterator)
                except StopIteration:
                    print(f"WARNING: Ran out of spawn locations for agent {agent_id}.")
                    continue

                self.simulation_state.add_component(
                    agent_id, PositionComponent(x=start_pos[0], y=start_pos[1])
                )
                self.simulation_state.add_component(
                    agent_id,
                    EnergyComponent(
                        current_energy=float(archetype["initial_energy"]),
                        initial_energy=float(archetype["initial_energy"]),
                    ),
                )
                self.simulation_state.add_component(
                    agent_id,
                    MetabolismComponent(
                        metabolic_rate=archetype["metabolic_rate"],
                        vision_range=archetype["vision_range"],
                    ),
                )
                self.simulation_state.add_component(
                    agent_id,
                    CommunicationComponent(message_range=7),
                )
                self.simulation_state.add_component(
                    agent_id,
                    PerceptionComponent(vision_range=archetype["vision_range"]),
                )
                self.simulation_state.add_component(
                    agent_id, TimeBudgetComponent(initial_time_budget=99999)
                )

                env.update_entity_position(agent_id, None, start_pos)
