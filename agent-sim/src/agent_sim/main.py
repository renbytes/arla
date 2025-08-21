import argparse
import importlib
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf


def app():
    """
    CLI entrypoint for running a single, local simulation.

    This function parses command-line arguments, loads the appropriate
    simulation configuration and entrypoint, and starts the simulation.
    It is designed for quick, direct runs without the Celery/MLflow stack.
    """
    # This allows the script to find the top-level 'simulations' directory.
    # The path is calculated relative to this file's location.
    # .../agent-sim/src/agent_sim/main.py -> .../agent-sim/src/ -> .../agent-sim/ -> project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    parser = argparse.ArgumentParser(
        description="ARLA Simulation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario", type=str, required=True, help="Path to the scenario JSON file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="simulations/soul_sim/config/base_config.yml",
        help="Path to the base simulation YAML config file.",
    )
    parser.add_argument(
        "--package",
        type=str,
        default="simulations.soul_sim",
        help="The Python package of the simulation to run (e.g., 'simulations.soul_sim').",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of simulation steps to run."
    )
    args = parser.parse_args()

    print("--- ARLA Local Simulation Runner ---")
    print(f"📦 Simulation Package: {args.package}")
    print(f"📄 Scenario File: {args.scenario}")
    print(f"⚙️  Base Config File: {args.config}")
    print(f"🏃 Simulation Steps: {args.steps}")
    print("----------------------------------")

    if not Path(args.scenario).exists():
        print(f"Error: Scenario file not found: {args.scenario}")
        exit(1)
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        exit(1)

    try:
        base_config = OmegaConf.load(args.config)
        overrides = OmegaConf.create(
            {
                "scenario_path": args.scenario,
                "simulation": {"steps": args.steps},
            }
        )
        final_config = OmegaConf.to_container(
            OmegaConf.merge(base_config, overrides), resolve=True
        )

        # Generate a valid UUID string for the run_id
        # run_id is set to None to signal MLflow to create a new run.
        run_id = None
        task_id = "local_run"
        # We'll create a dedicated experiment for all local runs of this type.
        experiment_name = f"{args.package}-local"

        sim_module = importlib.import_module(f"{args.package}.run")

        sim_module.start_simulation(
            run_id=run_id,
            task_id=task_id,
            experiment_name=experiment_name,
            config_overrides=final_config,
        )

        print("\n✅ Simulation completed successfully.")

    except ImportError:
        print(
            f"Error: Module '{args.package}.run' not found. Please check the package path."
        )
        exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred during the simulation: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    app()
