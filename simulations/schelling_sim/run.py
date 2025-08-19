# simulations/schelling_sim/run.py

import asyncio
from typing import Any, Dict, List, Optional

from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.core.ecs.component import ActionPlanComponent, Component
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_engine.simulation.engine import SimulationManager
from omegaconf import OmegaConf

from .actions import MoveToEmptyCellAction
from .components import PositionComponent, SchellingAgentComponent
from .environment import SchellingGridEnvironment
from .loader import SchellingScenarioLoader
from .systems import MovementSystem, SatisfactionSystem


# Helper classes to satisfy the SimulationManager's requirements
class SchellingActionGenerator(ActionGeneratorInterface):
    """Generates possible moves for unsatisfied agents."""

    def generate(self, sim_state, entity_id, tick) -> List[ActionPlanComponent]:
        move_action = MoveToEmptyCellAction()
        params_list = move_action.generate_possible_params(entity_id, sim_state, tick)
        return [ActionPlanComponent(action_type=move_action, params=p) for p in params_list]


class SchellingDecisionSelector(DecisionSelectorInterface):
    """A simple policy: if an agent can move, it will."""

    def select(self, sim_state, entity_id, actions) -> Optional[ActionPlanComponent]:
        return actions[0] if actions else None


class SchellingComponentFactory(ComponentFactoryInterface):
    """Creates component instances from saved data (not used in this simple run)."""

    def create_component(self, component_type: str, data: Dict[str, Any]) -> Component:
        if component_type.endswith("PositionComponent"):
            return PositionComponent(**data)
        if component_type.endswith("SchellingAgentComponent"):
            return SchellingAgentComponent(**data)
        raise TypeError(f"Unknown component type: {component_type}")


def start_simulation(
    run_id: str,
    task_id: str,
    experiment_id: str,
    config_overrides: Dict[str, Any],
) -> None:
    """The main entry point that sets up and runs the simulation."""
    config = OmegaConf.create(config_overrides)

    # This function is async because the underlying engine loop is.
    async def run_async():
        # 1. Initialize the world and loader
        # The environment is initialized by the loader, so we pass a placeholder here.
        env_placeholder = SchellingGridEnvironment(width=1, height=1)
        loader = SchellingScenarioLoader(
            simulation_state=None,  # The manager will create the state object
            scenario_path=config.scenario_path,
        )

        # 2. Instantiate the SimulationManager with our custom classes
        manager = SimulationManager(
            config=config,
            environment=env_placeholder,
            scenario_loader=loader,
            action_generator=SchellingActionGenerator(),
            decision_selector=SchellingDecisionSelector(),
            component_factory=SchellingComponentFactory(),
            db_logger=None,  # Not needed for this local run
            run_id=run_id,
            task_id=task_id,
            experiment_id=experiment_id,
        )

        # 3. Register the systems that contain our simulation's logic
        manager.register_system(SatisfactionSystem)
        manager.register_system(MovementSystem)

        # 4. Load the scenario to populate the world
        # We manually set the state on the loader after the manager creates it
        loader.simulation_state = manager.simulation_state
        loader.load()

        # 5. Run the main simulation loop
        print(f"🚀 Starting Schelling Simulation (Run ID: {run_id}) for {config.simulation.steps} steps...")
        await manager.run()
        print(f"✅ Simulation {run_id} completed.")

    asyncio.run(run_async())
