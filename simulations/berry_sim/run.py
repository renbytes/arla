# FILE: simulations/berry_sim/run.py

import asyncio
import importlib
import uuid
from typing import Any, Dict, Type

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

from .environment import BerryWorldEnvironment
from .loader import BerryScenarioLoader
from .metrics.causal_metrics_calculator import CausalMetricsCalculator
from .providers import (
    BerryPerceptionProvider,
    BerryRewardCalculator,
    BerryStateEncoder,
    BerryStateNodeEncoder,
)
from .systems import CausalMetricTrackerSystem, RenderingSystem


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
    """Initializes and runs the simulation logic."""
    config = OmegaConf.create(config_overrides)

    print(f"✅ Initializing simulation logic for MLflow run '{run_id}'.")

    db_manager = AsyncDatabaseManager()
    await db_manager.check_connection()

    run_uuid = uuid.UUID(run_id)
    db_emitter = DatabaseEmitter(db_manager=db_manager, simulation_id=run_uuid)
    mlflow_exporter = MLflowExporter()

    env_class = import_class(config.environment["class"])
    env: BerryWorldEnvironment = env_class(**config.environment.params)

    loader_class = import_class(config.scenario_loader["class"])
    action_generator_class = import_class(config.action_generator["class"])
    decision_selector_class = import_class(config.decision_selector["class"])
    component_factory_class = import_class(config.component_factory["class"])

    loader: BerryScenarioLoader = loader_class(
        simulation_state=None, scenario_path=config.scenario_path
    )

    manager = SimulationManager(
        config=config,
        environment=env,
        scenario_loader=loader,
        action_generator=action_generator_class(),
        decision_selector=decision_selector_class(simulation_state=None, config=config),
        component_factory=component_factory_class(),
        db_logger=db_manager,
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    # Connect providers to the manager's state
    loader.simulation_state = manager.simulation_state
    if hasattr(manager.decision_selector, "simulation_state"):
        manager.decision_selector.simulation_state = manager.simulation_state

    # Instantiate all providers
    state_encoder = BerryStateEncoder()
    state_node_encoder = BerryStateNodeEncoder(
        simulation_state=manager.simulation_state
    )
    perception_provider = BerryPerceptionProvider()
    reward_calc = BerryRewardCalculator()

    # Register all systems
    manager.register_system(ActionSystem, reward_calculator=reward_calc)
    manager.register_system(PerceptionSystem, perception_provider=perception_provider)

    metrics_calc = CausalMetricsCalculator()
    metric_tracker = CausalMetricTrackerSystem(
        manager.simulation_state,
        config,
        manager.cognitive_scaffold,
        calculator=metrics_calc,
    )
    manager.system_manager._systems.append(metric_tracker)

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

    for system_path in config.systems:
        system_class = import_class(system_path)
        if system_class is RenderingSystem and not config.rendering.get(
            "enabled", False
        ):
            continue
        manager.register_system(system_class)

    manager.register_system(
        MetricsSystem,
        calculators=[metrics_calc],
        exporters=[mlflow_exporter, db_emitter],
    )
    manager.register_system(LoggingSystem, exporters=[db_emitter])

    for action_path in config.actions:
        importlib.import_module(action_path)

    loader.load()

    if "QLearning" in config.decision_selector["class"]:
        for agent_id in manager.simulation_state.entities:
            if agent_id.startswith("agent_"):
                manager.simulation_state.add_component(
                    agent_id,
                    QLearningComponent(
                        state_feature_dim=14,
                        internal_state_dim=1,
                        action_feature_dim=5,
                        q_learning_alpha=config.learning.q_learning.alpha,
                        device=torch.device("cpu"),
                    ),
                )

    print(
        f"🚀 Starting Berry Simulation (Run ID: {run_id}) for {config.simulation.steps} steps..."
    )
    await manager.run()
    print(f"✅ Simulation {run_id} completed.")


def start_simulation(
    run_id: str, task_id: str, experiment_name: str, config_overrides: Dict[str, Any]
):
    """Entry point for external callers like Celery tasks."""
    # This function is now just a simple wrapper to run the async setup
    asyncio.run(setup_and_run(run_id, task_id, experiment_name, config_overrides))
