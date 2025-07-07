from celery_app import celery_app
from simulations.soul_sim.world import run_world


@celery_app.task(bind=True)
def execute_simulation(self, scenario_path: str) -> str:
    """Background task to run a simulation scenario."""
    return run_world(scenario_path)
