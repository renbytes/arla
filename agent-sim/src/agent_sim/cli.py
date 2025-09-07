import asyncio
import os
import subprocess
import sys
from pathlib import Path
from functools import wraps

import mlflow
import typer
from agent_sim.infrastructure.database.async_database_manager import (
    AsyncDatabaseManager,
)
from agent_sim.infrastructure.tasks.celery_app import app as celery_app
from agent_sim.infrastructure.tasks.simulation_tasks import run_experiment_task
from omegaconf import DictConfig, OmegaConf
from rich import print
from rich.console import Console
from rich.table import Table

# ------------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

EXPERIMENT_FILE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Path to the experiment definition YAML file.",
)

QUEUE_OPTION = typer.Option(
    "simulations",
    "--queue",
    "-Q",
    help="The Celery queue to consume from.",
)

CONCURRENCY_OPTION = typer.Option(
    4,
    "--concurrency",
    "-c",
    help="Number of concurrent worker processes.",
)

EXPERIMENTS_DIR_OPTION = typer.Option(
    str(PROJECT_ROOT / "experiments"),
    "--experiments-dir",
    help="The directory containing experiment definition files.",
    exists=True,
    file_okay=False,
)

app = typer.Typer(
    name="agentsim",
    help="A unified CLI for managing Agent-based Simulations.",
    add_completion=False,
)
console = Console()


# ------------------------------------------------------------------------
# Helper decorator for async commands
# ------------------------------------------------------------------------
def async_command(f):
    """Decorator to properly handle async functions in Typer commands."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# ------------------------------------------------------------------------
# Command: run-experiment
# ------------------------------------------------------------------------
@app.command(name="run-experiment")
@async_command
async def run_experiment(
    experiment_file: Path = EXPERIMENT_FILE_ARGUMENT,
):
    """Parse an experiment definition file, create a single parent experiment,
    and submit Celery tasks for each variation."""

    console.rule(f"[bold green]🚀 Launching Experiment: {experiment_file.name}")

    try:
        exp_def = OmegaConf.load(experiment_file)
    except Exception as e:
        print(
            f"[bold red]Error: Could not parse experiment file {experiment_file}.[/bold red]"
        )
        raise typer.Exit(code=1) from e

    if not isinstance(exp_def, DictConfig):
        print(
            f"[bold red]Error: Experiment file {experiment_file} must have a dictionary (mapping) at its root.[/bold red]"
        )
        raise typer.Exit(code=1)

    base_config_path = PROJECT_ROOT / exp_def.get("base_config_path", "")
    if not base_config_path.exists():
        print(
            f"[bold red]Error: Base config file not found at {base_config_path}[/bold red]"
        )
        raise typer.Exit(code=1)

    base_config = OmegaConf.load(base_config_path)
    variations = exp_def.get("variations", [{"name": "default", "overrides": {}}])
    experiment_name = exp_def.get("experiment_name", "UnnamedExp")

    # Create database manager and ensure connection
    console.print("🔬 Creating parent experiment records...")
    db_manager = AsyncDatabaseManager()

    try:
        await db_manager.check_connection()
    except Exception as e:
        console.print(
            f"[bold red]Error: Cannot connect to database. Error: {e}[/bold red]"
        )
        raise typer.Exit(code=1)

    try:
        # Set up MLflow tracking
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        else:
            console.print("[yellow]Warning: MLFLOW_TRACKING_URI not set[/yellow]")

        # Get or create MLflow experiment
        mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
        mlflow_exp_id = (
            mlflow_experiment.experiment_id
            if mlflow_experiment
            else mlflow.create_experiment(name=experiment_name)
        )

        # Calculate total runs
        scenarios = exp_def.get("scenarios", [])
        runs_per_scenario = exp_def.get("runs_per_scenario", 1)
        total_runs = len(scenarios) * runs_per_scenario * len(variations)

        # Create experiment in database
        db_experiment_uuid = await db_manager.create_experiment(
            name=experiment_name,
            config=OmegaConf.to_container(base_config, resolve=True),
            total_runs=total_runs,
            simulation_package=exp_def.get("simulation_package"),
            mlflow_experiment_id=mlflow_exp_id,
        )

        console.print(
            f"[green]✔ Parent DB record created: {db_experiment_uuid}[/green]"
        )
        console.print(f"[green]✔ Using MLflow experiment: {mlflow_exp_id}[/green]")

    except Exception as e:
        console.print(
            f"[bold red]FATAL: Could not create/get experiment records. Error: {e}[/bold red]"
        )
        raise typer.Exit(code=1)
    finally:
        # Clean up database connection if needed
        if hasattr(db_manager, "close"):
            await db_manager.close()

    # Submit tasks for each variation
    total_jobs = 0

    for variation in variations:
        if not isinstance(variation, (dict, DictConfig)):
            print(
                f"[yellow]Warning: Skipping invalid variation item in {experiment_file}.[/yellow]"
            )
            continue

        variation_name = variation.get("name", "unnamed_variation")
        variation_overrides = OmegaConf.to_container(variation.get("overrides", {}))

        # Merge base config with variation overrides
        final_config = OmegaConf.merge(base_config, variation.get("overrides", {}))
        config_dict = OmegaConf.to_container(final_config, resolve=True)

        console.print(
            f"\n[cyan]Submitting tasks for variation: [bold]{variation_name}[/bold]"
        )

        # Submit the task to Celery
        result = run_experiment_task.delay(
            scenario_paths=list(scenarios),
            runs_per_scenario=runs_per_scenario,
            base_config=config_dict,
            simulation_package=exp_def.get("simulation_package"),
            experiment_name=experiment_name,
            variation_name=variation_name,
            variation_overrides=variation_overrides,
            db_experiment_uuid=str(db_experiment_uuid),
            mlflow_exp_id=mlflow_exp_id,
        )

        jobs_in_variation = len(scenarios) * runs_per_scenario
        total_jobs += jobs_in_variation

        console.print(
            f"[green]✔ Submitted orchestration task for {jobs_in_variation} simulation runs.[/green]"
        )
        console.print(f"[dim]  Task ID: {result.id}[/dim]")

    # Final summary
    console.rule(f"[bold green]✅ Experiment '{experiment_name}' fully submitted.")
    print(f"Total simulation runs queued: [bold cyan]{total_jobs}[/bold cyan]")
    print("Monitor your Celery workers and MLflow UI for progress.")


# ------------------------------------------------------------------------
# Command: start-worker
# ------------------------------------------------------------------------
@app.command(name="start-worker")
def start_worker(
    queue: str = QUEUE_OPTION,
    concurrency: int = CONCURRENCY_OPTION,
):
    """Start a Celery worker with a simplified command."""
    console.rule(f"[bold blue]👷 Starting Celery Worker for Queue: {queue}")

    celery_app_path = "agent_sim.infrastructure.tasks.celery_app"
    command: list[str] = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        celery_app_path,
        "worker",
        "--loglevel=INFO",
        "-Q",
        queue,
        "-c",
        str(concurrency),
        "--pool=prefork",
    ]

    print(f"Running command: [dim]{' '.join(command)}[/dim]")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[bold red]Error starting Celery worker: {e}[/bold red]")
        print(
            "Please ensure Celery is installed in your environment and Redis is running."
        )
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        print(f"[bold red]Error: Python executable not found: {e}[/bold red]")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------------
# Command: list-experiments
# ------------------------------------------------------------------------
@app.command(name="list-experiments")
def list_experiments(
    experiments_dir: Path = EXPERIMENTS_DIR_OPTION,
):
    """Scan the experiments directory and list all available experiment files."""
    console.rule("[bold yellow]🔎 Available Experiments")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment File", style="dim", width=50)
    table.add_column("Full Path")

    # Find all YAML files
    exp_files = sorted(experiments_dir.glob("*.yaml")) + sorted(
        experiments_dir.glob("*.yml")
    )

    if not exp_files:
        print(
            f"[yellow]No experiment definition files (.yaml/.yml) found in:[/yellow] {experiments_dir}"
        )
        raise typer.Exit()

    for file_path in exp_files:
        table.add_row(file_path.name, str(file_path))

    console.print(table)


# ------------------------------------------------------------------------
# Command: health-check
# ------------------------------------------------------------------------
@app.command(name="health-check")
def health_check():
    """Ping the Celery workers to check if they are operational."""
    console.rule("[bold purple]🩺 Health Check")

    try:
        inspector = celery_app.control.inspect()
        active_workers = inspector.ping(timeout=5.0)

        if not active_workers:
            print("[bold red]Error: No active Celery workers found.[/bold red]")
            print(
                "Make sure your workers are running and connected to the same broker."
            )
            raise typer.Exit(code=1)

        print(
            f"[green]✔ Successfully pinged {len(active_workers)} active worker(s):[/green]"
        )
        for worker, reply in active_workers.items():
            print(f"  - [cyan]{worker}[/cyan]: {reply}")

    except Exception as e:
        print(f"[bold red]Error connecting to Celery broker: {e}[/bold red]")
        print("Please ensure Redis (or your broker) is running and accessible.")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------------
# Command: purge-queue
# ------------------------------------------------------------------------
@app.command(name="purge-queue")
def purge_queue(
    queue: str = QUEUE_OPTION,
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Purge all pending tasks from a specific queue."""
    console.rule(f"[bold red]⚠️  Purging Queue: {queue}")

    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to purge all tasks from the '{queue}' queue?",
            default=False,
        )
        if not confirm:
            print("[yellow]Purge cancelled.[/yellow]")
            raise typer.Exit()

    try:
        purged = celery_app.control.purge()
        print(
            f"[green]✔ Successfully purged {purged} task(s) from queue '{queue}'.[/green]"
        )
    except Exception as e:
        print(f"[bold red]Error purging queue: {e}[/bold red]")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------------
# Command: list-tasks
# ------------------------------------------------------------------------
@app.command(name="list-tasks")
def list_tasks():
    """List all active, scheduled, and reserved tasks."""
    console.rule("[bold cyan]📋 Task Status")

    try:
        inspector = celery_app.control.inspect()

        # Get active tasks
        active = inspector.active()
        if active:
            print("\n[bold green]Active Tasks:[/bold green]")
            for worker, tasks in active.items():
                print(f"  Worker: [cyan]{worker}[/cyan]")
                for task in tasks:
                    print(f"    - {task['name']} (ID: {task['id'][:8]}...)")
        else:
            print("[yellow]No active tasks.[/yellow]")

        # Get scheduled tasks
        scheduled = inspector.scheduled()
        if scheduled:
            print("\n[bold blue]Scheduled Tasks:[/bold blue]")
            for worker, tasks in scheduled.items():
                print(f"  Worker: [cyan]{worker}[/cyan]")
                for task in tasks:
                    print(
                        f"    - {task['request']['name']} (ID: {task['request']['id'][:8]}...)"
                    )

        # Get reserved tasks
        reserved = inspector.reserved()
        if reserved:
            print("\n[bold yellow]Reserved Tasks:[/bold yellow]")
            for worker, tasks in reserved.items():
                print(f"  Worker: [cyan]{worker}[/cyan]")
                for task in tasks:
                    print(f"    - {task['name']} (ID: {task['id'][:8]}...)")

    except Exception as e:
        print(f"[bold red]Error inspecting tasks: {e}[/bold red]")
        raise typer.Exit(code=1) from e


# ------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------
def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    main()
