# simulations/schelling_sim/run.py

import asyncio
import importlib
import uuid
from typing import Any, Dict, Type

from agent_engine.simulation.engine import SimulationManager
from agent_engine.systems.logging_system import LoggingSystem
from agent_engine.systems.metrics_system import MetricsSystem
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter
from omegaconf import OmegaConf

from .environment import SchellingGridEnvironment
from .loader import SchellingScenarioLoader
from .metrics.segregation_calculator import SegregationCalculator
from .providers import (
    SchellingActionGenerator,
    SchellingComponentFactory,
    SchellingDecisionSelector,
)


# A helper function to dynamically import system classes from the config
def import_class(class_path: str) -> Type:
    """Dynamically imports a class from its string path."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        print(f"[ERROR] Failed to import class at path '{class_path}': {e}")
        raise


def start_simulation(
    run_id: str,
    task_id: str,
    experiment_id: str,
    config_overrides: Dict[str, Any],
) -> None:
    """The main entry point that sets up and runs the simulation."""
    config = OmegaConf.create(config_overrides)

    async def run_async():
        db_manager = AsyncDatabaseManager()
        await db_manager.check_connection()

        run_uuid = uuid.UUID(run_id)
        database_emitter = DatabaseEmitter(db_manager=db_manager, simulation_id=run_uuid)
        mlflow_exporter = MLflowExporter(run_id=run_id, experiment_id=experiment_id)

        # --- FIX: Use the directly imported classes ---
        environment = SchellingGridEnvironment(**config.environment.params)
        loader = SchellingScenarioLoader(
            simulation_state=None,
            scenario_path=config.scenario_path,
        )
        # --- End of fix ---

        manager = SimulationManager(
            config=config,
            environment=environment,
            scenario_loader=loader,
            action_generator=SchellingActionGenerator(),
            decision_selector=SchellingDecisionSelector(),
            component_factory=SchellingComponentFactory(),
            db_logger=db_manager,
            run_id=run_id,
            task_id=task_id,
            experiment_id=experiment_id,
        )

        loader.simulation_state = manager.simulation_state

        segregation_calculator = SegregationCalculator()
        manager.register_system(
            MetricsSystem,
            calculators=[segregation_calculator],
            exporters=[mlflow_exporter, database_emitter],
        )
        manager.register_system(LoggingSystem, exporters=[database_emitter])

        # Systems are still loaded dynamically from the config for flexibility
        for system_path in config.systems:
            system_class = import_class(system_path)
            manager.register_system(system_class)

        for action_path in config.actions:
            importlib.import_module(action_path)

        loader.load()

        print(f"🚀 Starting Schelling Simulation (Run ID: {run_id}) for {config.simulation.steps} steps...")
        await manager.run()
        print(f"✅ Simulation {run_id} completed.")

    asyncio.run(run_async())
