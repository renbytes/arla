# simulations/disease_sim/run.py
"""
Main entry point for the Disease simulation.
"""

import asyncio
import importlib
import os
import uuid
from typing import Any, Dict, Type

import mlflow
import numpy as np
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

from .environment import DiseaseEnvironment
from .loader import DiseaseScenarioLoader
from .metrics.epidemiology_metrics_calculator import EpidemiologyMetricsCalculator
from .providers import (
    SmallWorldContactProvider,
    DiseaseActionGenerator,
    PassiveDecisionSelector,
    DiseaseComponentFactory,
    DiseaseRewardCalculator,
    DiseaseActionCostProvider,
)
from .systems import (
    DiseaseTransmissionSystem,
    DiseaseProgressionSystem,
    InterventionSystem,
)


def import_class(class_path: str) -> Type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def setup_and_run(
    run_id: str, task_id: str, experiment_id: str, config_overrides: Dict[str, Any]
):
    config = OmegaConf.create(config_overrides)
    db_manager = AsyncDatabaseManager()
    await db_manager.check_connection()

    run_uuid = uuid.UUID(run_id)
    db_emitter = DatabaseEmitter(db_manager=db_manager, simulation_id=run_uuid)
    mlflow_exporter = MLflowExporter()

    seed = config.simulation.get("random_seed")
    master_rng = (
        np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    )

    env_params = {
        "width": config.environment.params.width,
        "height": config.environment.params.height,
        "rng": master_rng,
    }
    env_class = import_class(config.environment["class"])
    env: DiseaseEnvironment = env_class(**env_params)

    loader = DiseaseScenarioLoader(
        simulation_state=None, scenario_path=config.scenario_path
    )

    manager = SimulationManager(
        config=config,
        environment=env,
        scenario_loader=loader,
        action_generator=DiseaseActionGenerator(),
        decision_selector=PassiveDecisionSelector(),
        component_factory=DiseaseComponentFactory(),
        db_logger=db_manager,
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    # Connect providers that need the live simulation_state
    loader.simulation_state = manager.simulation_state

    # Instantiate providers
    contact_provider = SmallWorldContactProvider(
        simulation_state=manager.simulation_state, config=config
    )
    reward_calculator = DiseaseRewardCalculator()
    action_cost_provider = DiseaseActionCostProvider()

    # Register all systems
    manager.register_system(
        ActionSystem,
        reward_calculator=reward_calculator,
        action_cost_provider=action_cost_provider,
    )
    manager.register_system(
        DiseaseTransmissionSystem, contact_provider=contact_provider
    )
    manager.register_system(DiseaseProgressionSystem)
    manager.register_system(InterventionSystem)

    # Register utility systems
    metrics_calculator = EpidemiologyMetricsCalculator()
    manager.register_system(
        MetricsSystem,
        calculators=[metrics_calculator],
        exporters=[mlflow_exporter, db_emitter],
    )
    manager.register_system(LoggingSystem, exporters=[db_emitter])

    # Load actions and scenario
    for action_path in config.actions:
        importlib.import_module(action_path)
    loader.load()

    # The contact provider network can only be built after the loader has created the agents
    contact_provider.build_network()

    print(
        f"🚀 Starting Disease Simulation (Run ID: {run_id}) for {config.simulation.steps} steps..."
    )
    await manager.run()
    print(f"✅ Simulation {run_id} completed.")


def start_simulation(
    run_id: str, task_id: str, experiment_name: str, config_overrides: Dict[str, Any]
):
    if not run_id:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        )
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else mlflow.create_experiment(name=experiment_name)
        )
        with mlflow.start_run(experiment_id=experiment_id) as run:
            current_run_id = run.info.run_id
            print(f"✅ Started new MLflow run for local execution: {current_run_id}")
            asyncio.run(
                setup_and_run(current_run_id, task_id, experiment_id, config_overrides)
            )
    else:
        asyncio.run(setup_and_run(run_id, task_id, experiment_name, config_overrides))
