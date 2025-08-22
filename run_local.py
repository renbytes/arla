# run_local.py
import argparse
import importlib
import traceback

from omegaconf import OmegaConf


def main():
    """A direct, root-level script to run local simulations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()

    print("--- ARLA Direct Local Runner ---")
    print(f"📦 Simulation Package: {args.package}")
    print(f"📄 Scenario File: {args.scenario}")
    print(f"⚙️  Base Config File: {args.config}")
    print(f"🏃 Simulation Steps: {args.steps}")
    print("----------------------------------")

    try:
        # Dynamically import the run module based on the package argument
        sim_module = importlib.import_module(f"{args.package}.run")

        # Load configs and run simulation
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

        print("\n🚀 Starting simulation...")
        sim_module.start_simulation(
            run_id=None,
            task_id="direct_local_run",
            experiment_name=f"{args.package}-direct-local",
            config_overrides=final_config,
        )
        print("\n✅ Simulation finished successfully.")

    except Exception as e:
        print(f"\n❌ An error occurred. Error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
