# simulations/schelling_sim/run.py

import asyncio
import importlib
import os
import uuid
from typing import Any, Dict, Type

import mlflow
from agent_engine.simulation.engine import SimulationManager
from agent_engine.systems.action_system import ActionSystem
from agent_engine.systems.logging_system import LoggingSystem
from agent_engine.systems.metrics_system import MetricsSystem
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter
from omegaconf import OmegaConf

from simulations.schelling_sim.systems import RenderingSystem

from .environment import SchellingGridEnvironment
from .loader import SchellingScenarioLoader
from .metrics.segregation_calculator import SegregationCalculator
from .providers import (
    SchellingActionGenerator,
    SchellingComponentFactory,
    SchellingDecisionSelector,
    SchellingRewardCalculator,
)


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
    experiment_name: str,
    config_overrides: Dict[str, Any],
) -> None:
    """The main entry point that sets up and runs the simulation."""
    config = OmegaConf.create(config_overrides)

    async def run_async():
        db_manager = AsyncDatabaseManager()
        await db_manager.check_connection()

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        )

        # This handles both local and cloud runs elegantly.
        # For local runs, run_id is a placeholder and a new run is created.
        # For Celery runs, the existing run_id is used to resume.
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"MLflow experiment '{experiment_name}' not found. Creating it.")
            experiment_id = mlflow.create_experiment(name=experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(
            run_id=run_id if run_id else None, experiment_id=experiment_id
        ) as run:
            # For local runs, we get the new ID from MLflow. For cloud runs, this will be the same as the input.
            current_run_id = run.info.run_id
            print(f"✅ MLflow run '{current_run_id}' started.")

            run_uuid = uuid.UUID(current_run_id)
            database_emitter = DatabaseEmitter(
                db_manager=db_manager, simulation_id=run_uuid
            )
            mlflow_exporter = MLflowExporter()

            environment = SchellingGridEnvironment(**config.environment.params)
            loader = SchellingScenarioLoader(
                simulation_state=None,
                scenario_path=config.scenario_path,
            )

            manager = SimulationManager(
                config=config,
                environment=environment,
                scenario_loader=loader,
                action_generator=SchellingActionGenerator(),
                decision_selector=SchellingDecisionSelector(),
                component_factory=SchellingComponentFactory(),
                db_logger=db_manager,
                run_id=current_run_id,
                task_id=task_id,
                experiment_id=experiment_id,
            )

            # Instantiate and register the rewards and ActionSystem
            # so that agents move to optimize their satisfaction through movement.
            reward_calculator = SchellingRewardCalculator()
            manager.register_system(ActionSystem, reward_calculator=reward_calculator)

            loader.simulation_state = manager.simulation_state

            segregation_calculator = SegregationCalculator()
            manager.register_system(
                MetricsSystem,
                calculators=[segregation_calculator],
                exporters=[mlflow_exporter, database_emitter],
            )
            manager.register_system(LoggingSystem, exporters=[database_emitter])

            if config.rendering.get("enabled", False):
                manager.register_system(RenderingSystem)

            for system_path in config.systems:
                system_class = import_class(system_path)
                manager.register_system(system_class)

            for action_path in config.actions:
                importlib.import_module(action_path)

            loader.load()

            print(
                f"🚀 Starting Schelling Simulation (Run ID: {current_run_id}) for {config.simulation.steps} steps..."
            )
            await manager.run()
            print(f"✅ Simulation {current_run_id} completed.")

    asyncio.run(run_async())
