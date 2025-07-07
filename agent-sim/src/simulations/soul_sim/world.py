"""World bootstrap that wires providers into the agent-engine SimulationManager."""

from pathlib import Path

# Stubs – replace with real imports from agent_engine and your providers
from agent_engine.simulation.engine import SimulationManager
from omegaconf import OmegaConf

from simulations.soul_sim.providers import (
    GridControllabilityProvider,
    GridVitalityProvider,
)


def run_world(scenario_path: str | None = None) -> str:
    # 1. Load YAML/JSON config
    cfg_file = Path(__file__).with_name("config.yaml")
    config = OmegaConf.load(cfg_file)

    # 2. Create provider instances
    controllability = GridControllabilityProvider()
    vitality = GridVitalityProvider()
    # TODO: Add this back in
    # state_node_encoder = GridStateNodeEncoder()

    # 3. Initialise SimulationManager (simplified)
    sim_mgr = SimulationManager(
        config=config,
        environment=None,  # Replace with your concrete EnvironmentInterface
        scenario_loader=lambda: None,
        action_generator=lambda *a, **kw: [],
        decision_selector=lambda *a, **kw: None,
    )

    # 4. Register cognitive systems with providers
    from agent_engine.systems.affect_system import AffectSystem

    sim_mgr.register_system(
        AffectSystem,
        vitality_metrics_provider=vitality,
        controllability_provider=controllability,
    )

    # Register other systems here…

    # 5. Run simulation (stub)
    sim_mgr.run()
    return "Simulation finished."
