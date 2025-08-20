# simulations/schelling_sim/loader.py

import json
import random
from typing import cast

from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface

from .components import GroupComponent, PositionComponent, SatisfactionComponent
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

        config = self.simulation_state.config
        num_agents = scenario_data.get("num_agents", 100)
        group_ratio = scenario_data.get("group_ratio", 0.5)
        satisfaction_threshold = config.simulation.get("satisfaction_threshold", 0.4)

        # The environment is now passed in, so we just ensure it's the correct type
        environment = cast(SchellingGridEnvironment, self.simulation_state.environment)
        if not environment:
            raise ValueError("Environment not initialized in SimulationState.")

        # Create agents
        locations = random.sample(environment.get_valid_positions(), num_agents)
        num_group_1 = int(num_agents * group_ratio)

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            agent_type = 1 if i < num_group_1 else 2
            position = locations[i]

            self.simulation_state.add_entity(agent_id)
            self.simulation_state.add_component(agent_id, PositionComponent(x=position[0], y=position[1]))

            self.simulation_state.add_component(agent_id, GroupComponent(agent_type=agent_type))
            self.simulation_state.add_component(
                agent_id, SatisfactionComponent(satisfaction_threshold=satisfaction_threshold)
            )

            self.simulation_state.add_component(agent_id, TimeBudgetComponent(initial_time_budget=1000))

        environment.initialize_from_state(self.simulation_state, self.simulation_state.entities)
