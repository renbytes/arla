# FILE: simulations/sugarscape_sim/run.py
"""
Main entry point for the Sugarscape simulation.
"""

import asyncio
import importlib
import os
import uuid
from typing import Any, Dict, Type

import mlflow
import numpy as np
import torch
from agent_engine.simulation.engine import SimulationManager
from agent_engine.systems.action_system import ActionSystem
from agent_engine.systems.causal_graph_system import CausalGraphSystem
from agent_engine.systems.components import QLearningComponent
from agent_engine.systems.logging_system import LoggingSystem
from agent_engine.systems.metrics_system import MetricsSystem
from agent_engine.systems.perception_system import PerceptionSystem
from agent_engine.systems.q_learning_system import QLearningSystem
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter
from omegaconf import OmegaConf

from .environment import SugarscapeEnvironment
from .metrics.sugarscape_metrics_calculator import SugarscapeMetricsCalculator
from .providers import (
    SugarscapeActionCostProvider,
    SugarscapePerceptionProvider,
    SugarscapeRewardCalculator,
    SugarscapeStateEncoder,
    SugarscapeStateNodeEncoder,
)
from .systems import (
    MetricTrackerSystem,
    RenderingSystem,
    VitalsSystem,  # NEW: Import the VitalsSystem
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


async def setup_and_run(
    run_id: str,
    task_id: str,
    experiment_id: str,
    config_overrides: Dict[str, Any],
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

    env_params = dict(config.environment.params)
    env_params["rng"] = master_rng
    env_class = import_class(config.environment["class"])
    env: SugarscapeEnvironment = env_class(**env_params)

    loader_class = import_class(config.scenario_loader["class"])
    action_generator_class = import_class(config.action_generator["class"])
    decision_selector_class = import_class(config.decision_selector["class"])
    component_factory_class = import_class(config.component_factory["class"])

    loader = loader_class(
        simulation_state=None, scenario_path=config.scenario_path, rng=master_rng
    )
    action_generator = action_generator_class(simulation_state=None, config=config)
    decision_selector = decision_selector_class(simulation_state=None, config=config)

    manager = SimulationManager(
        config=config,
        environment=env,
        scenario_loader=loader,
        action_generator=action_generator,
        decision_selector=decision_selector,
        component_factory=component_factory_class(),
        db_logger=db_manager,
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    loader.simulation_state = manager.simulation_state
    action_generator.simulation_state = manager.simulation_state
    decision_selector.simulation_state = manager.simulation_state

    # Instantiate all providers
    perception_provider = SugarscapePerceptionProvider()
    reward_calculator = SugarscapeRewardCalculator(
        simulation_state=manager.simulation_state, config=config
    )
    state_encoder = SugarscapeStateEncoder()
    state_node_encoder = SugarscapeStateNodeEncoder(
        simulation_state=manager.simulation_state
    )
    action_cost_provider = SugarscapeActionCostProvider()
    sugarscape_calculator = SugarscapeMetricsCalculator()

    for system_path in config.systems:
        system_class = import_class(system_path)
        if system_class is RenderingSystem and not config.rendering.get(
            "enabled", False
        ):
            continue
        manager.register_system(system_class)

    # Register engine systems with all required dependencies
    manager.register_system(
        ActionSystem,
        reward_calculator=reward_calculator,
        action_cost_provider=action_cost_provider,
    )
    manager.register_system(PerceptionSystem, perception_provider=perception_provider)
    causal_system = CausalGraphSystem(
        manager.simulation_state,
        config,
        manager.cognitive_scaffold,
        state_node_encoder=state_node_encoder,
    )
    manager.system_manager._systems.append(causal_system)
    manager.register_system(
        QLearningSystem,
        state_encoder=state_encoder,
        causal_graph_system=causal_system,
    )

    # NEW: Register the VitalsSystem to handle agent death
    manager.register_system(VitalsSystem)

    manager.register_system(MetricTrackerSystem, calculator=sugarscape_calculator)
    manager.register_system(
        MetricsSystem,
        calculators=[sugarscape_calculator],
        exporters=[mlflow_exporter, db_emitter],
    )
    manager.register_system(LoggingSystem, exporters=[db_emitter])

    for action_path in config.actions:
        importlib.import_module(action_path)
    loader.load()

    if "QLearningDecisionSelector" in config.decision_selector["class"]:
        for agent_id in manager.simulation_state.entities:
            if "agent_" in agent_id or "forager_" in agent_id or "rusher_" in agent_id:
                manager.simulation_state.add_component(
                    agent_id,
                    QLearningComponent(
                        state_feature_dim=9,
                        internal_state_dim=1,
                        action_feature_dim=6,
                        q_learning_alpha=config.learning.q_learning.alpha,
                        device=torch.device("cpu"),
                    ),
                )

    print(
        f"🚀 Starting Sugarscape Simulation (Run ID: {run_id}) for {config.simulation.steps} steps..."
    )
    await manager.run()
    print(f"✅ Simulation {run_id} completed.")


def start_simulation(
    run_id: str, task_id: str, experiment_name: str, config_overrides: Dict[str, Any]
):
    """Synchronous entry point for external callers."""
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
