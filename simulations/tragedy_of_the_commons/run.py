"""
Main entry point for the Tragedy of the Commons simulation.

This script orchestrates the setup and execution of the simulation. It
initializes the SimulationManager, registers all necessary systems (both
world-specific and from the agent-engine), injects provider implementations,
loads the scenario, and starts the main simulation loop.
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
from agent_engine.systems.components import QLearningComponent
from agent_engine.systems.logging_system import LoggingSystem
from agent_engine.systems.metrics_system import MetricsSystem
from agent_engine.systems.q_learning_system import QLearningSystem
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter
from omegaconf import OmegaConf

from .environment import CommonsEnvironment
from .loader import CommonsScenarioLoader
from .metrics.metrics_calculator import CommonsMetricsCalculator
from .providers import (
    CommonsActionCostProvider,
    CommonsActionGenerator,
    CommonsComponentFactory,
    CommonsRewardCalculator,
    CommonsStateEncoder,
    QLearningDecisionSelector,
)
from .systems import (
    GrazingSystem,
    MetabolismSystem,
    MovementSystem,
    RenderingSystem,
    ResourceRegenerationSystem,
    VitalsSystem,
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
    """Initializes and runs the full simulation logic."""
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
    env: CommonsEnvironment = env_class(**env_params)

    decision_selector_class = import_class(config.decision_selector["class"])

    loader = CommonsScenarioLoader(
        simulation_state=None, scenario_path=config.scenario_path, rng=master_rng
    )
    action_generator = CommonsActionGenerator()
    decision_selector = decision_selector_class(simulation_state=None, config=config)

    manager = SimulationManager(
        config=config,
        environment=env,
        scenario_loader=loader,
        action_generator=action_generator,
        decision_selector=decision_selector,
        component_factory=CommonsComponentFactory(),
        db_logger=db_manager,
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    # Connect providers to the manager's state
    loader.simulation_state = manager.simulation_state
    decision_selector.simulation_state = manager.simulation_state
    env.simulation_state = manager.simulation_state

    # Instantiate providers
    state_encoder = CommonsStateEncoder()
    reward_calculator = CommonsRewardCalculator()
    action_cost_provider = CommonsActionCostProvider()

    # Register world-specific systems
    manager.register_system(MetabolismSystem)
    manager.register_system(VitalsSystem)
    manager.register_system(MovementSystem)
    manager.register_system(GrazingSystem)
    manager.register_system(ResourceRegenerationSystem)

    # Register engine systems
    manager.register_system(
        ActionSystem,
        reward_calculator=reward_calculator,
        action_cost_provider=action_cost_provider,
    )
    manager.register_system(
        QLearningSystem, state_encoder=state_encoder, causal_graph_system=None
    )

    # Register utility systems
    metrics_calculator = CommonsMetricsCalculator()
    manager.register_system(
        MetricsSystem,
        calculators=[metrics_calculator],
        exporters=[mlflow_exporter, db_emitter],
    )
    manager.register_system(LoggingSystem, exporters=[db_emitter])

    if config.rendering.get("enabled", False):
        manager.register_system(RenderingSystem)

    # Load actions and scenario
    for action_path in config.actions:
        importlib.import_module(action_path)
    loader.load()

    # Add QLearningComponent if using the Q-learning selector
    if decision_selector_class is QLearningDecisionSelector:
        for agent_id in manager.simulation_state.entities:
            if agent_id.startswith("herder_"):
                manager.simulation_state.add_component(
                    agent_id,
                    QLearningComponent(
                        state_feature_dim=6,  # energy + 5 resource patches
                        internal_state_dim=1,
                        action_feature_dim=3,  # move, graze, wait
                        q_learning_alpha=config.learning.q_learning.alpha,
                        device=torch.device("cpu"),
                    ),
                )

    print(
        f"🚀 Starting simulation (Run ID: {run_id}) for {config.simulation.steps} steps..."
    )
    await manager.run()
    print(f"✅ Simulation {run_id} completed.")


def start_simulation(
    run_id: str, task_id: str, experiment_name: str, config_overrides: Dict[str, Any]
):
    """Synchronous entry point for external callers like Celery tasks."""
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
            asyncio.run(
                setup_and_run(current_run_id, task_id, experiment_id, config_overrides)
            )
    else:
        asyncio.run(setup_and_run(run_id, task_id, experiment_name, config_overrides))
