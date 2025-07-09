import asyncio
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

# Generic imports from the monorepo
from agent_core.agents.actions.base_action import Action
from agent_engine.simulation.engine import SimulationManager
from agent_engine.systems.action_system import ActionSystem
from agent_engine.systems.affect_system import AffectSystem
from agent_engine.systems.causal_graph_system import CausalGraphSystem
from agent_engine.systems.goal_system import GoalSystem
from agent_engine.systems.identity_system import IdentitySystem
from agent_engine.systems.q_learning_system import QLearningSystem
from agent_engine.systems.reflection_system import ReflectionSystem

# Imports specific to this simulation (soul-sim)
from .config.schemas import SoulSimAppConfig
from .environment.grid_world import GridWorld
from .providers import (
    SoulSimActionGenerator,
    SoulSimControllabilityProvider,
    SoulSimDecisionSelector,
    SoulSimNarrativeContextProvider,
    SoulSimRewardCalculator,
    SoulSimStateEncoder,
    SoulSimStateNodeEncoder,
    SoulSimVitalityMetricsProvider,
)
from .simulation.scenario_loader import ScenarioLoader
from .systems.combat_system import CombatSystem
from .systems.movement_system import MovementSystem
from .systems.resource_system import ResourceSystem


def start_simulation(run_id: str, task_id: str, experiment_id: str, config_overrides: Dict[str, Any]):
    """
    The main entry point for running a soul-sim simulation.
    This function is called by the generic Celery task.
    """
    # This synchronous wrapper calls the main async function
    asyncio.run(setup_and_run(run_id, task_id, experiment_id, config_overrides))


async def setup_and_run(run_id: str, task_id: str, experiment_id: str, config_overrides: Dict[str, Any]):
    """Asynchronous setup and execution for the simulation."""

    print(f"--- [{task_id}] Initializing Soul-Sim ---")

    # 1. Create the final, validated configuration
    base_config = OmegaConf.load("simulations/soul_sim/config/base_config.yml")
    final_config: DictConfig = OmegaConf.merge(base_config, config_overrides)

    try:
        config_dict = OmegaConf.to_container(final_config, resolve=True)
        if isinstance(config_dict, dict):
            SoulSimAppConfig(**config_dict)
        else:
            raise TypeError("Resolved config is not a dictionary.")
        print(f"[{task_id}] Configuration validated successfully.")
    except Exception as e:
        print(f"[{task_id}] FATAL: Configuration validation failed for soul-sim: {e}")
        raise

    # 2. Initialize world-specific singletons and providers
    Action.initialize_action_registry()

    env_config = OmegaConf.to_container(final_config.environment, resolve=True)
    grid_size = env_config.get('grid_world_size', [50, 50])
    environment = GridWorld(width=grid_size[0], height=grid_size[1])

    # Instantiate the ScenarioLoader. It only needs the config at this stage.
    scenario_loader_instance = ScenarioLoader(final_config)

    providers = {
        "action_generator": SoulSimActionGenerator(),
        "decision_selector": SoulSimDecisionSelector(),
        "reward_calculator": SoulSimRewardCalculator(),
        "state_encoder": SoulSimStateEncoder(),
        "narrative_context_provider": SoulSimNarrativeContextProvider(),
        "controllability_provider": SoulSimControllabilityProvider(),
        "state_node_encoder": SoulSimStateNodeEncoder(),
        "vitality_metrics_provider": SoulSimVitalityMetricsProvider(),
    }

    # 3. Create the SimulationManager. It will create the SimulationState internally.
    manager = SimulationManager(
        config=final_config,
        environment=environment,
        scenario_loader=scenario_loader_instance,
        action_generator=providers["action_generator"],
        decision_selector=providers["decision_selector"],
        run_id=run_id,
        task_id=task_id,
        experiment_id=experiment_id,
    )

    # 4. ARCHITECTURAL FIX: Manually link the state and load the scenario.
    # The manager creates the state, but this entry point script is responsible
    # for telling the loader to populate it.
    scenario_loader_instance.simulation_state = manager.simulation_state
    scenario_loader_instance.load() # This now calls the load() method on your loader.

    # 5. Register all systems
    manager.register_system(QLearningSystem, state_encoder=providers["state_encoder"])
    manager.register_system(ActionSystem, reward_calculator=providers["reward_calculator"])
    manager.register_system(
        AffectSystem,
        vitality_metrics_provider=providers["vitality_metrics_provider"],
        controllability_provider=providers["controllability_provider"],
    )
    manager.register_system(
        CausalGraphSystem,
        state_node_encoder=providers["state_node_encoder"],
    )
    manager.register_system(
        ReflectionSystem,
        narrative_context_provider=providers["narrative_context_provider"],
    )
    manager.register_system(IdentitySystem)
    manager.register_system(GoalSystem)
    manager.register_system(MovementSystem)
    manager.register_system(ResourceSystem)
    manager.register_system(CombatSystem)

    # 6. Run the simulation
    print(f"--- [{task_id}] Starting simulation loop for run: {run_id} ---")
    await manager.run()
    print(f"--- [{task_id}] Simulation loop finished for run: {run_id} ---")
