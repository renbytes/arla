# FILE: src/agent_engine/persistence/snapshot_manager.py

from agent_persist.models import AgentSnapshot, ComponentSnapshot, SimulationSnapshot

from agent_engine.simulation.simulation_state import SimulationState


def create_snapshot_from_state(sim_state: SimulationState) -> SimulationSnapshot:
    """
    Converts a live SimulationState object into a serializable SimulationSnapshot model.
    """
    agent_snapshots = []
    for agent_id, components in sim_state.entities.items():
        component_snapshots = []
        for component in components.values():
            # Use the component's class name and its to_dict() method
            comp_snapshot = ComponentSnapshot(
                component_type=component.__class__.__name__,
                data=component.to_dict(),
            )
            component_snapshots.append(comp_snapshot)

        agent_snapshots.append(AgentSnapshot(agent_id=agent_id, components=component_snapshots))

    return SimulationSnapshot(
        simulation_id=sim_state.simulation_id,
        current_tick=sim_state.current_tick,  # Assuming current_tick is on sim_state
        agents=agent_snapshots,
        environment_state=sim_state.environment.to_dict() if sim_state.environment else None,
    )
