# simulations/berry_sim/run.py

import asyncio
import importlib
import os
import uuid
from typing import Any, Dict, Type

import mlflow
from omegaconf import OmegaConf

from agent_engine.simulation.engine import SimulationManager
from agent_engine.systems.action_system import ActionSystem
from agent_engine.systems.causal_graph_system import CausalGraphSystem
from agent_engine.systems.logging_system import LoggingSystem
from agent_engine.systems.metrics_system import MetricsSystem
from agent_engine.systems.q_learning_system import QLearningSystem
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter

from .environment import BerryWorldEnvironment
from .loader import BerryScenarioLoader
from .metrics.causal_metrics_calculator import CausalMetricsCalculator
from .providers import (
    BerryActionGenerator,
    BerryComponentFactory,
    BerryDecisionSelector,
    BerryRewardCalculator,
    BerryStateEncoder,
    BerryStateNodeEncoder,
)
from .systems import CausalMetricTrackerSystem


def import_class(class_path: str) -> Type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def start_simulation(
    run_id: str, task_id: str, experiment_name: str, config_overrides: Dict[str, Any]
):
    config = OmegaConf.create(config_overrides)

    async def run_async():
        db_manager = AsyncDatabaseManager()
        await db_manager.check_connection()
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        )

        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else mlflow.create_experiment(name=experiment_name)
        )

        with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as run:
            current_run_id = run.info.run_id
            print(f"✅ MLflow run '{current_run_id}' started.")

            run_uuid = uuid.UUID(current_run_id)
            db_emitter = DatabaseEmitter(db_manager=db_manager, simulation_id=run_uuid)
            mlflow_exporter = MLflowExporter()

            env = BerryWorldEnvironment(**config.environment.params)
            loader = BerryScenarioLoader(
                simulation_state=None, scenario_path=config.scenario_path
            )

            manager = SimulationManager(
                config=config,
                environment=env,
                scenario_loader=loader,
                action_generator=BerryActionGenerator(),
                decision_selector=BerryDecisionSelector(),
                component_factory=BerryComponentFactory(),
                db_logger=db_manager,
                run_id=current_run_id,
                task_id=task_id,
                experiment_id=experiment_id,
            )
            loader.simulation_state = manager.simulation_state

            # --- Instantiate Providers that need state ---
            state_encoder = BerryStateEncoder(simulation_state=manager.simulation_state)
            state_node_encoder = BerryStateNodeEncoder(
                simulation_state=manager.simulation_state
            )

            # --- Instantiate Metrics Calculators and Trackers ---
            metrics_calc = CausalMetricsCalculator()
            metric_tracker = CausalMetricTrackerSystem(
                manager.simulation_state,
                config,
                manager.cognitive_scaffold,
                calculator=metrics_calc,
            )
            # Manually add the tracker to the system manager
            manager.system_manager._systems.append(metric_tracker)

            # --- Register Core Systems ---
            reward_calc = BerryRewardCalculator()
            manager.register_system(ActionSystem, reward_calculator=reward_calc)

            causal_system = None
            if config.simulation.get("enable_causal_system", False):
                print("INFO: CausalGraphSystem is ENABLED for this run.")
                causal_system = CausalGraphSystem(
                    manager.simulation_state,
                    config,
                    manager.cognitive_scaffold,
                    state_node_encoder=state_node_encoder,
                )
                manager.system_manager._systems.append(causal_system)
            else:
                print("INFO: CausalGraphSystem is DISABLED for this run.")

            manager.register_system(
                QLearningSystem,
                state_encoder=state_encoder,
                causal_graph_system=causal_system,
            )

            # --- Register Simulation-Specific and Logging Systems ---
            for system_path in config.systems:
                manager.register_system(import_class(system_path))

            manager.register_system(
                MetricsSystem,
                calculators=[metrics_calc],
                exporters=[mlflow_exporter, db_emitter],
            )
            manager.register_system(LoggingSystem, exporters=[db_emitter])

            for action_path in config.actions:
                importlib.import_module(action_path)

            loader.load()
            print(
                f"🚀 Starting Berry Simulation (Run ID: {current_run_id}) for {config.simulation.steps} steps..."
            )
            await manager.run()
            print(f"✅ Simulation {current_run_id} completed.")

    asyncio.run(run_async())
