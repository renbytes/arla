# src/agent_engine/simulation/engine.py
"""
A world-agnostic simulation engine that orchestrates the main ECS loop.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, cast

import numpy as np

# Imports from agent_core
from agent_core.agents.action_generator_interface import ActionGeneratorInterface
from agent_core.agents.decision_selector_interface import DecisionSelectorInterface
from agent_core.cognition.scaffolding import CognitiveScaffold
from agent_core.core.ecs.component import TimeBudgetComponent
from agent_core.core.ecs.event_bus import EventBus
from agent_core.environment.interface import EnvironmentInterface
from agent_core.simulation.scenario_loader_interface import ScenarioLoaderInterface

# Imports from agent-persist
from agent_persist.store import FileStateStore
from omegaconf import DictConfig, OmegaConf

# Imports from agent-engine
from agent_engine.persistence.snapshot_manager import create_snapshot_from_state
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
        config: DictConfig,
        environment: EnvironmentInterface,
        scenario_loader: ScenarioLoaderInterface,
        action_generator: ActionGeneratorInterface,
        decision_selector: DecisionSelectorInterface,
        task_id: str = "local_run",
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.config: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))
        self.device = "cpu"  # Simplified for the engine
        self.save_path = Path(config.get("persistence", {}).get("save_path", "./sim_snapshots"))

        # --- Injected Dependencies ---
        self.environment = environment
        self.scenario_loader = scenario_loader
        self.action_generator = action_generator
        self.decision_selector = decision_selector
        # ---

        self._setup_simulation_ids(run_id, task_id, experiment_id)
        self._setup_rng()

        # Initialize core services and state
        self.event_bus = EventBus(self.config)
        self.simulation_state = SimulationState(self.config, self.device)
        self.cognitive_scaffold = CognitiveScaffold(self.simulation_id, self.config)

        self.system_manager = SystemManager(self.simulation_state, self.config, self.cognitive_scaffold)

        self._initialize_state()

        print("Engine: Manager initialized. World will be populated by the scenario loader.")

    def register_system(self, system_class: Type[Any], **kwargs: Any) -> None:
        """A convenience method to register a system with the SystemManager."""
        self.system_manager.register_system(system_class, **kwargs)

    # Async:The main run loop is now async.
    async def run(self) -> None:
        """Executes the main ECS simulation loop asynchronously."""
        num_steps = self.config.get("simulation", {}).get("steps", 100)
        print(f"\nStarting simulation {self.simulation_id} for {num_steps} steps...")

        for step in range(num_steps):
            print(f"\n--- Simulation Step {step + 1}/{num_steps} ---")
            self.simulation_state.current_tick = step

            # --- 1. PROCESS ENTITY TURNS FIRST ---
            active_entities: List[str] = []
            for eid, comps in self.simulation_state.entities.items():
                time_comp = comps.get(TimeBudgetComponent)
                if isinstance(time_comp, TimeBudgetComponent) and time_comp.is_active:
                    active_entities.append(eid)

            if not active_entities:
                print("All entities are inactive. Ending simulation.")
                break

            self.main_rng.shuffle(active_entities)

            for entity_id in active_entities:
                self._process_entity_turn(entity_id, step)

            # --- 2. UPDATE ALL SYSTEMS ONCE ---
            # (The first call to update_all has been removed)
            await self.system_manager.update_all(current_tick=step)

            # --- 3. PERIODICALLY SAVE STATE ---
            # Save state after systems have been updated for the tick
            if step > 0 and step % 50 == 0:
                self.save_state(step)

        # --- 4. EXECUTE THESE ACTIONS *AFTER* THE LOOP FINISHES ---
        print("\nSimulation loop finished.")
        self.save_state(num_steps)  # Final save

    def save_state(self, tick: int) -> None:
        """Saves the current simulation state to a file."""
        print(f"--- Saving state at tick {tick} ---")
        snapshot = create_snapshot_from_state(self.simulation_state)
        filepath = self.save_path / self.simulation_id / f"snapshot_tick_{tick}.json"
        store = FileStateStore(filepath)
        store.save(snapshot)

    def load_state(self, filepath: str) -> None:
        """Loads a simulation state from a file."""
        # This is the more complex part you'll implement later
        print(f"--- Loading state from {filepath} ---")
        # store = FileStateStore(filepath)
        # snapshot = store.load()
        # self.simulation_state = restore_state_from_snapshot(snapshot)
        pass

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
        seed = self.config.get("simulation", {}).get("random_seed")
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
