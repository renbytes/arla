# src/infrastructure/tasks/simulation_tasks.py

import os
import uuid
from typing import Any, Dict, Optional, cast

import mlflow  # type: ignore
from celery import current_task  # type: ignore
from config.schemas import AppConfig
from omegaconf import DictConfig, OmegaConf  # type: ignore
from pydantic import ValidationError  # type: ignore
from src.agents.actions.base_action import Action
from src.core.simulation.engine import SimulationManager
from src.data.async_database_manager import AsyncDatabaseManager
from src.data.async_runner import async_runner
from src.infrastructure.tasks.celery_app import app

# Set the tracking URI if it's not already set in the environment.
# This points our script to the MLflow server container.
if not os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("http://mlflow:5000")


@app.task(bind=True, name="tasks.run_simulation", queue="simulations")
def run_simulation_task(
    self,
    scenario_path: str,
    config_overrides: Dict[str, Any],
    run_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single simulation as a Celery task, with MLflow tracking.
    """
    try:
        AppConfig(**config_overrides)
        print("--- Configuration successfully validated by Pydantic ---")
    except ValidationError as e:
        print("--- CONFIGURATION ERROR ---")
        print(e)
        raise

    Action.initialize_action_registry()

    if run_id is None:
        run_id = str(uuid.uuid4())

    task_id = current_task.request.id
    print(f"[{task_id}] Starting simulation run: {run_id} for experiment '{experiment_name}'")

    base_dir = f"data/tasks/{task_id}" if not os.getenv("CLOUD_RUN") else f"/tmp/tasks/{task_id}"
    log_dir = f"{base_dir}/logs"
    db_dir = f"{base_dir}/db"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    self.update_state(state="PROGRESS", meta={"status": "Initializing simulation", "progress": 0, "run_id": run_id})

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_id):
        try:
            config: DictConfig = OmegaConf.create(config_overrides)

            temp_config = OmegaConf.create({"simulation": {"log_directory": log_dir, "database_directory": db_dir}})

            # FIX: Cast the result of OmegaConf.merge to DictConfig to satisfy mypy.
            # We know the result will be a DictConfig because we are merging two dict-like configs.
            config = cast(DictConfig, OmegaConf.merge(config, temp_config))

            mlflow.log_params(config_overrides)
            mlflow.set_tag("scenario", os.path.basename(scenario_path).replace(".json", ""))
            if experiment_id:
                mlflow.set_tag("experiment_id", experiment_id)
            if task_id:
                mlflow.set_tag("celery_task_id", task_id)

            self.update_state(state="PROGRESS", meta={"status": "Running simulation", "progress": 10, "run_id": run_id})

            manager = SimulationManager(
                run_id=run_id,
                config=config,
                scenario_path=scenario_path,
                task_id=task_id,
                experiment_id=experiment_id,
            )
            manager.run()

            self.update_state(state="PROGRESS", meta={"status": "Analyzing results", "progress": 90, "run_id": run_id})

            output_txt_path = os.path.join(log_dir, "final_output_multi_entity.txt")
            if os.path.exists(output_txt_path):
                mlflow.log_artifact(output_txt_path, "outputs")

            output_gif_path = os.path.join(log_dir, "simulation_output.gif")
            if os.path.exists(output_gif_path):
                mlflow.log_artifact(output_gif_path, "visualizations")

            final_config_dict = OmegaConf.to_container(config, resolve=True)

            results = {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "simulation_id": manager.simulation_id,
                "task_id": task_id,
                "scenario_path": scenario_path,
                "config": final_config_dict,
                "status": "completed",
                "results_path": base_dir,
                "log_path": log_dir,
            }

            print(f"[{task_id}] Completed simulation run: {run_id}")
            return results

        except Exception as exc:
            error_msg = f"Simulation failed: {str(exc)}"
            print(f"[{task_id}] ERROR: {error_msg}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", error_msg)

            if any(keyword in str(exc).lower() for keyword in ["database is locked", "timeout", "connection"]):
                print(f"[{task_id}] Retrying due to transient error...")
                raise self.retry(exc=exc, countdown=60, max_retries=3)  # noqa: B904

            raise


@app.task(bind=True, name="tasks.run_experiment", queue="experiments")
def run_experiment_task(
    self,
    scenario_paths: list,
    runs_per_scenario: int,
    base_config: Dict[str, Any],
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit multiple simulation tasks for a complete experiment.
    """
    experiment_id = str(uuid.uuid4())

    if experiment_name is None:
        experiment_name = f"experiment_{experiment_id[:8]}"

    print(f"Starting experiment: {experiment_name} (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)

    db = None
    try:
        db = AsyncDatabaseManager()
        create_exp_coro = db.create_experiment(
            experiment_id=experiment_id,
            name=experiment_name,
            config=base_config,
            total_runs=len(scenario_paths) * runs_per_scenario,
        )
        async_runner.run_async(create_exp_coro)
        print(f"Successfully created and committed experiment record for ID: {experiment_id}")
    except Exception as e:
        print(f"CRITICAL: Failed to log experiment to database: {e}. Aborting task.")
        self.update_state(state="FAILURE", meta={"exc_type": type(e).__name__, "exc_message": str(e)})
        raise
    finally:
        if db:
            db.close()

    job_ids = []

    for scenario_path in scenario_paths:
        print(f"DEBUG: Processing scenario in experiment task: {scenario_path}")
        scenario_name = os.path.basename(scenario_path).replace(".json", "")

        for run_num in range(runs_per_scenario):
            config = base_config.copy()
            config["random_seed"] = config.get("random_seed", 1) + run_num

            job = run_simulation_task.delay(
                scenario_path=scenario_path,
                config_overrides=config,
                run_id=f"{experiment_id}_{scenario_name}_{run_num:03d}",
                experiment_id=experiment_id,
                experiment_name=experiment_name,
            )
            job_ids.append(job.id)
            print(f"   Submitted job {job.id} for {scenario_name} run {run_num}")

    result = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "total_jobs": len(job_ids),
        "job_ids": job_ids,
        "scenarios": scenario_paths,
        "runs_per_scenario": runs_per_scenario,
        "status": "submitted",
    }

    print(f"Experiment {experiment_name} submitted with {len(job_ids)} jobs")
    return result


@app.task(name="tasks.health_check")
def health_check():
    """Simple health check task for monitoring."""
    return {"status": "healthy", "message": "Worker is operational"}
