# simulations/schelling_sim/loader.py

import json
import random
from typing import cast

from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface

from .components import PositionComponent, SchellingAgentComponent
from .environment import SchellingGridEnvironment


class SchellingScenarioLoader(ScenarioLoaderInterface):
    """
    Loads a scenario for the Schelling segregation model.

    This loader reads a JSON scenario file to configure the simulation grid
    and populate it with agents of different types at random locations.
    """

    def __init__(self, simulation_state, scenario_path: str):
        self.simulation_state = simulation_state
        self.scenario_path = scenario_path

    def load(self) -> None:
        """
        Loads the scenario, initializes the environment, and creates agents.
        """
        with open(self.scenario_path, "r") as f:
            scenario_data = json.load(f)

        # --- FIX: Use the grid dimensions from the scenario ---
        grid_width = scenario_data.get("grid_width", 50)
        grid_height = scenario_data.get("grid_height", 50)
        num_agents = scenario_data.get("num_agents", 100)
        group_ratio = scenario_data.get("group_ratio", 0.5)
        satisfaction_threshold = scenario_data.get("satisfaction_threshold", 0.4)

        # Initialize the environment with the correct dimensions
        self.simulation_state.environment = SchellingGridEnvironment(width=grid_width, height=grid_height)
        environment = cast(SchellingGridEnvironment, self.simulation_state.environment)

        # Create agents
        locations = random.sample(environment.get_valid_positions(), num_agents)
        num_group_1 = int(num_agents * group_ratio)

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            agent_type = 1 if i < num_group_1 else 2
            position = locations[i]

            self.simulation_state.add_entity(agent_id)
            self.simulation_state.add_component(agent_id, PositionComponent(x=position[0], y=position[1]))
            self.simulation_state.add_component(
                agent_id,
                SchellingAgentComponent(
                    agent_type=agent_type,
                    satisfaction_threshold=satisfaction_threshold,
                ),
            )
            # Add a basic time budget component for compatibility
            self.simulation_state.add_component(agent_id, TimeBudgetComponent(initial_time_budget=1000))

        # Populate the environment grid from the final component state
        environment.initialize_from_state(self.simulation_state, self.simulation_state.entities)
