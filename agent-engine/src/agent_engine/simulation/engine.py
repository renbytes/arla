# src/agent_engine/simulation/engine.py
"""
A world-agnostic simulation engine that orchestrates the main ECS loop.
"""

import time
import uuid
from pathlib import Path
from typing import Any, List, Optional, Type

import numpy as np

# Imports from agent_core
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.cognition.scaffolding import CognitiveScaffold
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.core.ecs.component_factory_interface import ComponentFactoryInterface
from agent_core.core.ecs.event_bus import EventBus
from agent_core.environment.interface import EnvironmentInterface
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface
from agent_persist.store import FileStateStore

# Imports from agent-engine
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import SystemManager


class SimulationManager:
    """
    This manager is responsible for stepping through time, processing entity
    decisions, and updating all registered systems. It is decoupled from any
    specific world implementation or game rules.
    """

    def __init__(
        self,
        config: Any,  # Now accepts a validated Pydantic model
        environment: EnvironmentInterface,
        scenario_loader: ScenarioLoaderInterface,
        action_generator: ActionGeneratorInterface,
        decision_selector: DecisionSelectorInterface,
        component_factory: ComponentFactoryInterface,
        db_logger: Any,
        task_id: str = "local_run",
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        # The config is now a Pydantic model, not a dict
        self.config = config
        self.device = "cpu"
        # Use direct attribute access on the validated config object
        self.save_path = Path(self.config.simulation.log_directory) / "snapshots"
        self.db_logger = db_logger

        # Injected Dependencies
        self.environment = environment
        self.scenario_loader = scenario_loader
        self.action_generator = action_generator
        self.decision_selector = decision_selector
        self.component_factory = component_factory

        self._setup_simulation_ids(run_id, task_id, experiment_id)
        self._setup_rng()

        # Initialize core services and state, passing the validated config object
        self.event_bus = EventBus(self.config)
        self.simulation_state = SimulationState(self.config, self.device)
        self.cognitive_scaffold = CognitiveScaffold(self.simulation_id, self.config, db_logger=self.db_logger)

        self.system_manager = SystemManager(self.simulation_state, self.config, self.cognitive_scaffold)

        self._initialize_state()

        print("Engine: Manager initialized. World will be populated by the scenario loader.")

    def register_system(self, system_class: Type[Any], **kwargs: Any) -> None:
        """A convenience method to register a system with the SystemManager."""
        self.system_manager.register_system(system_class, **kwargs)

    async def run(self, start_step: int = 0, end_step: Optional[int] = None) -> None:
        """
        Executes the main ECS simulation loop asynchronously.
        Args:
            start_step: The tick to start the simulation from (for resuming).
            end_step: The tick to end the simulation on. If None, runs for the
                      number of steps specified in the config.
        """
        if end_step is None:
            # Use direct attribute access
            num_steps = self.config.simulation.steps
        else:
            num_steps = end_step

        print(f"\nStarting simulation {self.simulation_id} from step {start_step} to {num_steps}...")

        for step in range(start_step, num_steps):
            print(f"\n--- Simulation Step {step + 1}/{num_steps}")
            self.simulation_state.current_tick = step

            # 1. CHECK FOR ACTIVE ENTITIES (EXIT EARLY IF NONE)
            active_entities: List[str] = []
            for eid, comps in self.simulation_state.entities.items():
                time_comp = comps.get(TimeBudgetComponent)
                if isinstance(time_comp, TimeBudgetComponent) and time_comp.is_active:
                    active_entities.append(eid)

            if not active_entities:
                print("All entities are inactive. Ending simulation.")
                break

            # 2. UPDATE ALL SYSTEMS ONCE (e.g., for state caching)
            await self.system_manager.update_all(current_tick=step)

            # 3. PROCESS ENTITY TURNS
            if self.main_rng:
                self.main_rng.shuffle(active_entities)

            for entity_id in active_entities:
                self._process_entity_turn(entity_id, step)

            # 4. PERIODICALLY SAVE STATE
            if step > 0 and step % 50 == 0:
                self.save_state(step)

        # 5. EXECUTE THESE ACTIONS *AFTER* THE LOOP FINISHES
        print("\nSimulation loop finished.")
        self.save_state(num_steps)

    def save_state(self, tick: int) -> None:
        """Saves the current simulation state to a file."""
        print(f"--- Saving state at tick {tick}")
        snapshot = self.simulation_state.to_snapshot()
        filepath = self.save_path / self.simulation_id / f"snapshot_tick_{tick}.json"
        store = FileStateStore(filepath)
        store.save(snapshot)

    def load_state(self, filepath: str) -> None:
        """
        Loads the entire simulation state from a checkpoint file.
        This replaces the existing self.simulation_state with a new one.
        """
        print(f"--- Loading state from {filepath}")
        store = FileStateStore(Path(filepath))
        snapshot = store.load()

        # Re-create the simulation state using the class method
        self.simulation_state = SimulationState.from_snapshot(
            snapshot=snapshot,
            config=self.config,
            component_factory=self.component_factory,
            environment=self.environment,
            event_bus=self.event_bus,
            db_logger=self.db_logger,
        )

        print(f"--- State successfully loaded. Resuming at tick {self.simulation_state.current_tick + 1}")

    def _process_entity_turn(self, entity_id: str, current_tick: int) -> None:
        """Handles the decision-making and action-dispatching for a single entity."""
        time_comp = self.simulation_state.get_component(entity_id, TimeBudgetComponent)
        if not time_comp or not hasattr(time_comp, "is_active") or not time_comp.is_active:
            return

        print(f"   Processing turn for {entity_id}...")

        possible_actions = self.action_generator.generate(self.simulation_state, entity_id, current_tick)
        if not possible_actions:
            print(f"   {entity_id} has no possible actions and passes the turn.")
            return

        chosen_plan = self.decision_selector.select(self.simulation_state, entity_id, possible_actions)
        if not chosen_plan:
            return

        self.simulation_state.add_component(entity_id, chosen_plan)

        if self.event_bus:
            self.event_bus.publish(
                "action_chosen",
                {
                    "entity_id": entity_id,
                    "action_plan_component": chosen_plan,
                    "current_tick": current_tick,
                },
            )

    def _setup_simulation_ids(self, run_id: Optional[str], task_id: str, experiment_id: Optional[str]) -> None:
        """Sets up unique IDs for the simulation run."""
        if run_id:
            self.simulation_id = run_id
        else:
            timestamp = int(time.time() * 1000)
            self.simulation_id = f"sim_{timestamp}_{uuid.uuid4().hex[:8]}"

        self.task_id = task_id
        self.experiment_id = experiment_id

    def _setup_rng(self) -> None:
        """Initializes random number generators."""
        # Use direct attribute access
        seed = self.config.simulation.random_seed
        if seed is not None:
            self.main_rng = np.random.default_rng(seed)
        else:
            self.main_rng = np.random.default_rng()

    def _initialize_state(self) -> None:
        """Initializes the simulation state with core services."""
        self.simulation_state.simulation_id = self.simulation_id
        self.simulation_state.environment = self.environment
        self.simulation_state.event_bus = self.event_bus
        self.simulation_state.system_manager = self.system_manager
        self.simulation_state.cognitive_scaffold = self.cognitive_scaffold
        self.simulation_state.main_rng = self.main_rng
        self.simulation_state.db_logger = self.db_logger
