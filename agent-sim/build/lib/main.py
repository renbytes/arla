"""Entry-point for running the simulation.

In real usage you would parse CLI flags, load scenario YAML/JSON,
instantiate the SimulationManager, inject providers, then call run().
"""
from simulations.soul_sim.world import run_world

if __name__ == "__main__":
    run_world()
