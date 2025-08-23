# simulations/berry_sim/loader.py

import json
import random
from typing import cast

from agent_core.core.ecs.component import PerceptionComponent, TimeBudgetComponent
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface

from .components import (
    HealthComponent,
    PositionComponent,
    WaterComponent,
    RockComponent,
)
from .environment import BerryWorldEnvironment


class BerryScenarioLoader(ScenarioLoaderInterface):
    """Loads the Berry Toxicity experiment scenario."""

    def __init__(self, simulation_state, scenario_path: str):
        self.simulation_state = simulation_state
        self.scenario_path = scenario_path

    def load(self) -> None:
        """Loads the scenario, initializes the environment, and creates agents."""
        with open(self.scenario_path, "r") as f:
            scenario_data = json.load(f)

        env = cast(BerryWorldEnvironment, self.simulation_state.environment)
        if not env:
            raise ValueError("Environment not initialized in SimulationState.")

        # Place water sources
        num_water_sources = scenario_data.get("num_water_sources", 10)
        placed_water = 0
        while placed_water < num_water_sources:
            pos = (random.randint(0, env.width - 1), random.randint(0, env.height - 1))
            if pos in env.water_locations:
                continue

            env.water_locations.add(pos)
            entity_id = f"water_{pos[0]}_{pos[1]}"
            self.simulation_state.add_entity(entity_id)
            self.simulation_state.add_component(entity_id, WaterComponent())
            placed_water += 1

        # Place rock formations
        for _ in range(scenario_data.get("num_rock_formations", 20)):
            pos = env.get_random_empty_cell()
            # CORRECTED: Check if a valid position was found before using it.
            if pos:
                env.rock_locations.add(pos)
                entity_id = f"rock_{pos[0]}_{pos[1]}"
                self.simulation_state.add_entity(entity_id)
                self.simulation_state.add_component(entity_id, RockComponent())

        # Create agents
        num_agents = scenario_data.get("num_agents", 100)
        config = self.simulation_state.config
        initial_health = config.agent.vitals.initial_health
        vision_range = config.agent.get(
            "vision_range", 7
        )  # Get vision range from config

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.simulation_state.add_entity(agent_id)

            pos = env.get_random_empty_cell()
            if pos:
                self.simulation_state.add_component(
                    agent_id, PositionComponent(x=pos[0], y=pos[1])
                )
                self.simulation_state.add_component(
                    agent_id, HealthComponent(initial_health, initial_health)
                )
                self.simulation_state.add_component(
                    agent_id, TimeBudgetComponent(initial_time_budget=2000)
                )
                # NEW: Add the PerceptionComponent to each agent
                self.simulation_state.add_component(
                    agent_id, PerceptionComponent(vision_range=vision_range)
                )
                env.add_entity(agent_id, pos)
            else:
                print(f"Warning: Could not find empty cell for agent {agent_id}")
