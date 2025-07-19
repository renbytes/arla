# simulations/emergence_sim/run.py
import asyncio
import os
import uuid
from typing import Any, Dict, Optional

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
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.logging.database_emitter import DatabaseEmitter
from agent_sim.infrastructure.logging.mlflow_exporter import MLflowExporter
from omegaconf import OmegaConf

from simulations.emergence_sim.config.schemas import EmergenceSimAppConfig
from simulations.emergence_sim.environment.world import EmergenceEnvironment
from simulations.emergence_sim.metrics.calculators import EmergenceMetricsCalculator
from simulations.emergence_sim.providers.action_providers import (
    EmergenceActionGenerator,
    EmergenceDecisionSelector,
)
from simulations.emergence_sim.providers.simulation_providers import (
    EmergenceControllabilityProvider,
    EmergenceNarrativeContextProvider,
    EmergenceRewardCalculator,
    EmergenceStateEncoder,
    EmergenceVitalityMetricsProvider,
)
from simulations.emergence_sim.simulation.component_factory import (
    EmergenceComponentFactory,
)
from simulations.emergence_sim.simulation.scenario_loader import (
    EmergenceScenarioLoader,
)
from simulations.emergence_sim.systems.decay_system import DecaySystem
from simulations.emergence_sim.systems.move_system import MovementSystem
from simulations.emergence_sim.systems.narrative_consensus_system import (
    NarrativeConsensusSystem,
)
from simulations.emergence_sim.systems.normative_abstraction_system import (
    NormativeAbstractionSystem,
)
from simulations.emergence_sim.systems.object_interaction_system import (
    ObjectInteractionSystem,
)
from simulations.emergence_sim.systems.ritualization_system import RitualizationSystem
from simulations.emergence_sim.systems.social_credit_system import SocialCreditSystem
from simulations.emergence_sim.systems.symbol_negotiation_system import (
    SymbolNegotiationSystem,
)
from simulations.emergence_sim.systems.synergy_system import SynergySystem


def start_simulation(run_id: str, task_id: str, experiment_id: str, config_overrides: Dict[str, Any]):
    """
    The main entry point for running an emergence-sim simulation.
    This function is called by the generic Celery task.
    """
    asyncio.run(setup_and_run(run_id, task_id, experiment_id, config_overrides))


async def setup_and_run(
    run_id: str,
    task_id: str,
    experiment_id: str,
    config_overrides: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
):
    """Asynchronous setup and execution for the simulation."""

    print(f"--- [{task_id}] Initializing Emergence-Sim")

    # 1. Load, merge, and validate the final configuration
    try:
        base_config = OmegaConf.load("simulations/emergence_sim/config/emergence_config.yml")
        final_config_dict = OmegaConf.to_container(OmegaConf.merge(base_config, config_overrides), resolve=True)
        if not isinstance(final_config_dict, dict):
            raise TypeError("Resolved config is not a dictionary.")
        config = EmergenceSimAppConfig(**final_config_dict)
        print(f"[{task_id}] Configuration validated successfully.")
    except Exception as e:
        print(f"[{task_id}] FATAL: Configuration validation failed for emergence-sim: {e}")
        raise

    # 2. Initialize world-specific singletons and providers
    action_registry.load_actions_from_paths(config.action_modules)
    db_manager = AsyncDatabaseManager()
    database_emitter = DatabaseEmitter(db_manager=db_manager, simulation_id=uuid.UUID(run_id))
    mlflow_exporter = MLflowExporter()
    environment = EmergenceEnvironment(
        width=config.environment.grid_world_size[0],
        height=config.environment.grid_world_size[1],
        num_objects=config.environment.num_objects,
    )
    component_factory = EmergenceComponentFactory(environment=environment, config=config)
    state_encoder = EmergenceStateEncoder()

    providers = {
        "action_generator": EmergenceActionGenerator(),
        "decision_selector": EmergenceDecisionSelector(state_encoder=state_encoder),
        "reward_calculator": EmergenceRewardCalculator(),
        "state_encoder": state_encoder,
        "narrative_context_provider": EmergenceNarrativeContextProvider(),
        "controllability_provider": EmergenceControllabilityProvider(),
        "state_node_encoder": EmergenceNarrativeContextProvider(),
        "vitality_metrics_provider": EmergenceVitalityMetricsProvider(),
    }

    # 3. Create the generic SimulationManager
    manager = SimulationManager(
        config=config,
        environment=environment,
        scenario_loader=None,  # Temporarily None
        action_generator=providers["action_generator"],
        decision_selector=providers["decision_selector"],
        component_factory=component_factory,
        db_logger=db_manager,  # ARGUMENT ADDED HERE
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    # 4. Load initial state
    starting_tick = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        await manager.load_state(checkpoint_path)
        starting_tick = manager.simulation_state.current_tick + 1
    else:
        # Create the loader here, passing the manager's state object
        scenario_loader_instance = EmergenceScenarioLoader(config, environment, manager.simulation_state)
        # Link the loader back to the manager and load the data
        manager.scenario_loader = scenario_loader_instance
        manager.scenario_loader.load()

    # 5. Register all systems with the manager
    # Register systems with inter-dependencies in the correct order
    manager.register_system(CausalGraphSystem, state_node_encoder=providers["state_node_encoder"])
    causal_system_instance = manager.system_manager.get_system(CausalGraphSystem)

    manager.register_system(
        QLearningSystem,
        state_encoder=providers["state_encoder"],
        causal_graph_system=causal_system_instance,
    )

    # Register the rest of the core systems
    manager.register_system(ActionSystem, reward_calculator=providers["reward_calculator"])
    manager.register_system(
        AffectSystem,
        vitality_metrics_provider=providers["vitality_metrics_provider"],
        controllability_provider=providers["controllability_provider"],
    )
    manager.register_system(
        ReflectionSystem,
        narrative_context_provider=providers["narrative_context_provider"],
    )
    manager.register_system(IdentitySystem)
    manager.register_system(GoalSystem)

    # Logging and Metrics Systems
    metrics_calculator = EmergenceMetricsCalculator()
    manager.register_system(LoggingSystem, exporters=[database_emitter, mlflow_exporter])
    manager.register_system(
        MetricsSystem,
        calculators=[metrics_calculator],
        exporters=[database_emitter],
    )
    manager.register_system(DecaySystem)
    manager.register_system(MovementSystem)
    manager.register_system(ObjectInteractionSystem)
    manager.register_system(SynergySystem)

    # Conditionally register custom simulation systems
    if config.simulation.systems.enable_symbol_negotiation:
        manager.register_system(SymbolNegotiationSystem)
    if config.simulation.systems.enable_narrative_consensus:
        manager.register_system(NarrativeConsensusSystem)
    if config.simulation.systems.enable_ritualization:
        manager.register_system(RitualizationSystem)
    if config.simulation.systems.enable_social_credit:
        manager.register_system(SocialCreditSystem)
    if config.simulation.systems.enable_normative_abstraction:
        manager.register_system(NormativeAbstractionSystem)

    # 6. Run the simulation
    print(f"--- [{task_id}] Starting simulation loop for run: {run_id} from tick {starting_tick}")
    await manager.run(start_step=starting_tick)
    print(f"--- [{task_id}] Simulation loop finished for run: {run_id}")
