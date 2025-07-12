# agent-persist/src/agent_persist/restore.py

from agent_core.environment.interface import EnvironmentInterface
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.utils.class_importer import import_class
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from omegaconf import DictConfig, OmegaConf

from agent_persist.models import SimulationSnapshot


def restore_state_from_snapshot(
    snapshot: SimulationSnapshot,
    config: DictConfig,
    environment: EnvironmentInterface,
    db_logger: AsyncDatabaseManager,
) -> SimulationState:
    """Reconstructs a live SimulationState from a Pydantic snapshot model."""

    # 1. Initialize a new, empty SimulationState
    # Convert OmegaConf to a standard dict and provide a valid device string.
    sim_state = SimulationState(config=OmegaConf.to_container(config, resolve=True), device="cpu")
    sim_state.current_tick = snapshot.current_tick
    sim_state.simulation_id = snapshot.simulation_id
    # Assign the passed environment and logger to the new state object.
    sim_state.environment = environment
    sim_state.db_logger = db_logger

    # The import_class helper is called inside the loop, so this line is not needed.
    # component_importer = import_class()

    # 2. Iterate through the agents in the snapshot
    for agent_snapshot in snapshot.agents:
        agent_id = agent_snapshot.agent_id
        sim_state.add_entity(agent_id)

        # 3. Iterate through the components for each agent
        for comp_snapshot in agent_snapshot.components:
            try:
                # Use the reusable utility function here, passing the component type path.
                component_class = import_class(comp_snapshot.component_type)

                # Instantiate the component using its saved data dictionary
                component_instance = component_class(**comp_snapshot.data)
                sim_state.add_component(agent_id, component_instance)
            except Exception as e:
                print(f"Could not restore component {comp_snapshot.component_type} for agent {agent_id}: {e}")

    # 4. Restore the environment state if it exists
    # This check is now safe because sim_state.environment is guaranteed to be assigned.
    if snapshot.environment_state and sim_state.environment:
        sim_state.environment.restore_from_dict(snapshot.environment_state)

    return sim_state
