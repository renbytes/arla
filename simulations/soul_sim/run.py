# FILE: simulations/soul_sim/run.py
# VERB: Replaces the entire file

import asyncio
import os
import uuid
from typing import Any, Dict, Optional

# Core/Engine Imports
from agent_core.agents.actions.action_registry import action_registry
from agent_engine.simulation.engine import SimulationManager
from agent_engine.systems.action_system import ActionSystem
from agent_engine.systems.affect_system import AffectSystem
from agent_engine.systems.causal_graph_system import CausalGraphSystem
from agent_engine.systems.goal_system import GoalSystem
from agent_engine.systems.identity_system import IdentitySystem
from agent_engine.systems.logging_system import LoggingSystem
from agent_engine.systems.metrics_system import MetricsSystem
from agent_engine.systems.q_learning_system import QLearningSystem
from agent_engine.systems.reflection_system import ReflectionSystem

# Infrastructure Imports
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter
from omegaconf import DictConfig, OmegaConf

# Simulation-Specific Imports
from simulations.soul_sim.component_factory import SoulSimComponentFactory
from simulations.soul_sim.config.schemas import SoulSimAppConfig
from simulations.soul_sim.environment.grid_world import GridWorld
from simulations.soul_sim.metrics.vitals_calculator import VitalsAndEconomyCalculator
from simulations.soul_sim.providers import (
    SoulSimActionGenerator,
    SoulSimControllabilityProvider,
    SoulSimDecisionSelector,
    SoulSimNarrativeContextProvider,
    SoulSimRewardCalculator,
    SoulSimStateEncoder,
    SoulSimStateNodeEncoder,
    SoulSimVitalityMetricsProvider,
)
from simulations.soul_sim.simulation.scenario_loader import ScenarioLoader
from simulations.soul_sim.systems.combat_system import CombatSystem
from simulations.soul_sim.systems.decay_system import DecaySystem
from simulations.soul_sim.systems.failed_states_system import FailedStatesSystem
from simulations.soul_sim.systems.movement_system import MovementSystem
from simulations.soul_sim.systems.nest_system import NestSystem
from simulations.soul_sim.systems.render_system import RenderSystem
from simulations.soul_sim.systems.resource_system import ResourceSystem
from simulations.soul_sim.systems.social_interaction_system import (
    SocialInteractionSystem,
)


def start_simulation(run_id: str, task_id: str, experiment_id: str, config_overrides: Dict[str, Any]):
    """
    The main entry point for running a soul-sim simulation.
    This function is called by the generic Celery task.
    """
    # This synchronous wrapper calls the main async function
    asyncio.run(setup_and_run(run_id, task_id, experiment_id, config_overrides))


async def setup_and_run(
    run_id: str,
    task_id: str,
    experiment_id: str,
    config_overrides: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
):
    """Asynchronous setup and execution for the simulation."""

    print(f"--- [{task_id}] Initializing Soul-Sim ---")

    # 1. Load and validate the final configuration
    base_config = OmegaConf.load("simulations/soul_sim/config/base_config.yml")
    final_config: DictConfig = OmegaConf.merge(base_config, config_overrides)

    try:
        config_dict = OmegaConf.to_container(final_config, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError("Resolved config is not a dictionary.")
        SoulSimAppConfig(**config_dict)
        print(f"[{task_id}] Configuration validated successfully.")
    except Exception as e:
        print(f"[{task_id}] FATAL: Configuration validation failed for soul-sim: {e}")
        raise

    # 2. Initialize world-specific singletons and providers
    action_registry.load_actions_from_paths(config_dict.get("action_modules", []))

    db_manager = AsyncDatabaseManager()
    database_emitter = DatabaseEmitter(db_manager=db_manager, simulation_id=uuid.UUID(run_id))
    mlflow_exporter = MLflowExporter()

    env_config = config_dict.get("environment", {})
    grid_size = env_config.get("grid_world_size", (50, 50))
    environment = GridWorld(width=grid_size[0], height=grid_size[1])

    # Instantiate all providers needed by the systems
    providers = {
        "action_generator": SoulSimActionGenerator(),
        "decision_selector": SoulSimDecisionSelector(),
        "reward_calculator": SoulSimRewardCalculator(config=config_dict),
        "state_encoder": SoulSimStateEncoder(),
        "narrative_context_provider": SoulSimNarrativeContextProvider(),
        "controllability_provider": SoulSimControllabilityProvider(),
        "state_node_encoder": SoulSimStateNodeEncoder(),
        "vitality_metrics_provider": SoulSimVitalityMetricsProvider(),
    }

    # Instantiate the ScenarioLoader and ComponentFactory
    scenario_loader_instance = ScenarioLoader(config_dict)
    component_factory = SoulSimComponentFactory(environment=environment, config=config_dict)

    # 3. Create the generic SimulationManager from the agent-engine
    manager = SimulationManager(
        config=final_config,
        environment=environment,
        scenario_loader=scenario_loader_instance,
        action_generator=providers["action_generator"],
        decision_selector=providers["decision_selector"],
        component_factory=component_factory,
        db_logger=db_manager,
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    # 4. Load initial state (either from checkpoint or scenario file)
    starting_tick = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        manager.load_state(checkpoint_path)
        starting_tick = manager.simulation_state.current_tick + 1
    else:
        # Inject the created simulation_state into the loader before calling load
        scenario_loader_instance.simulation_state = manager.simulation_state
        scenario_loader_instance.load()

    # 5. Register all systems with the manager
    # Core Cognitive Systems
    manager.register_system(QLearningSystem, state_encoder=providers["state_encoder"])
    manager.register_system(ActionSystem, reward_calculator=providers["reward_calculator"])
    manager.register_system(
        AffectSystem,
        vitality_metrics_provider=providers["vitality_metrics_provider"],
        controllability_provider=providers["controllability_provider"],
    )
    manager.register_system(CausalGraphSystem, state_node_encoder=providers["state_node_encoder"])
    manager.register_system(
        ReflectionSystem,
        narrative_context_provider=providers["narrative_context_provider"],
    )
    manager.register_system(IdentitySystem)
    manager.register_system(GoalSystem)

    # World-Specific Systems
    manager.register_system(MovementSystem)
    manager.register_system(ResourceSystem)
    manager.register_system(CombatSystem)
    manager.register_system(DecaySystem)
    manager.register_system(FailedStatesSystem)
    manager.register_system(NestSystem)
    manager.register_system(SocialInteractionSystem)

    # Logging and Metrics Systems
    vitals_calculator = VitalsAndEconomyCalculator()
    manager.register_system(LoggingSystem, exporters=[database_emitter, mlflow_exporter])
    manager.register_system(
        MetricsSystem,
        calculators=[vitals_calculator],
        exporters=[database_emitter, mlflow_exporter],
    )
    manager.register_system(RenderSystem)  # Add the render system

    # 6. Run the simulation
    print(f"--- [{task_id}] Starting simulation loop for run: {run_id} from tick {starting_tick} ---")
    await manager.run(start_step=starting_tick)
    print(f"--- [{task_id}] Simulation loop finished for run: {run_id} ---")

    # Finalize any systems that need it (like rendering the GIF)
    if render_system := manager.system_manager.get_system(RenderSystem):
        render_system.finalize()
