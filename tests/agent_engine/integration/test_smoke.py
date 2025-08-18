# agent-engine/tests/integration/test_smoke.py

import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    TimeBudgetComponent,
)
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.environment.interface import EnvironmentInterface
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface
from agent_engine.simulation.engine import SimulationManager

# --- Mock Implementations for the Smoke Test ---


class MockMoveAction(ActionInterface):
    @property
    def action_id(self) -> str:
        return "move"

    @property
    def name(self) -> str:
        return "Move"

    def get_base_cost(self, simulation_state: Any) -> float:
        return 10.0  # A fixed cost for determinism

    def generate_possible_params(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return [{}]  # Always one possible move

    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        # In a real scenario, this would be handled by a world system
        pass

    def get_feature_vector(self, *args, **kwargs) -> List[float]:
        return [1.0]


class MockActionGenerator(ActionGeneratorInterface):
    def generate(self, *args, **kwargs) -> List[ActionPlanComponent]:
        return [ActionPlanComponent(action_type=MockMoveAction())]


class MockDecisionSelector(DecisionSelectorInterface):
    def select(
        self, simulation_state: Any, entity_id: str, possible_actions: List[ActionPlanComponent]
    ) -> Optional[ActionPlanComponent]:
        # Always select the first action for deterministic behavior
        return possible_actions[0] if possible_actions else None


class MockScenarioLoader(ScenarioLoaderInterface):
    def __init__(self, sim_manager):
        self._sim_manager = sim_manager

    def load(self) -> None:
        # Create two agents with a starting time budget
        state = self._sim_manager.simulation_state
        state.add_entity("agent_1")
        state.add_component("agent_1", TimeBudgetComponent(initial_time_budget=100.0))
        state.add_entity("agent_2")
        state.add_component("agent_2", TimeBudgetComponent(initial_time_budget=100.0))


class MockComponentFactory(ComponentFactoryInterface):
    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        # A real factory would dynamically import classes
        if component_type.endswith("TimeBudgetComponent"):
            return TimeBudgetComponent(**data)
        raise TypeError(f"Unknown component type for factory: {component_type}")


@pytest.mark.asyncio
class TestIntegrationSmoke(unittest.TestCase):
    """
    A simple integration "smoke test" that runs a minimal, deterministic
    scenario from end-to-end to ensure the core simulation loop works.
    """

    async def test_run_minimal_simulation(self):
        """
        Runs a 5-tick simulation with two agents that repeatedly perform
        a 'move' action with a fixed cost, and asserts the final state.
        """
        # 1. Arrange: Set up the full SimulationManager with mock implementations
        mock_config = MagicMock()
        mock_config.simulation.steps = 5
        mock_config.simulation.log_directory = "logs"
        mock_config.simulation.random_seed = 42  # for determinism

        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_db_logger = MagicMock()

        # The scenario loader needs a reference to the manager to populate its state
        manager = SimulationManager(
            config=mock_config,
            environment=mock_env,
            scenario_loader=None,  # Will be set after instantiation
            action_generator=MockActionGenerator(),
            decision_selector=MockDecisionSelector(),
            component_factory=MockComponentFactory(),
            db_logger=mock_db_logger,
        )
        # Now inject the manager into the loader
        manager.scenario_loader = MockScenarioLoader(manager)

        # Manually load the scenario to set up the initial state
        manager.scenario_loader.load()

        # 2. Act: Run the simulation loop
        await manager.run()

        # 3. Assert: Check if the final state is as expected
        final_state = manager.simulation_state

        # Each agent runs 5 times, each action costs 10.0
        # Expected budget = 100.0 - (5 * 10.0) = 50.0
        agent1_time_comp = final_state.get_component("agent_1", TimeBudgetComponent)
        agent2_time_comp = final_state.get_component("agent_2", TimeBudgetComponent)

        self.assertIsNotNone(agent1_time_comp)
        self.assertIsNotNone(agent2_time_comp)

        self.assertAlmostEqual(agent1_time_comp.current_time_budget, 50.0)
        self.assertAlmostEqual(agent2_time_comp.current_time_budget, 50.0)

        # Verify the simulation ran for the correct number of ticks
        self.assertEqual(final_state.current_tick, 4)  # Ticks are 0-indexed


if __name__ == "__main__":
    pytest.main()
