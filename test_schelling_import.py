# test_schelling_import.py
import sys
from pathlib import Path

print("--- Direct Schelling Import Test ---")

# Print diagnostics to be certain about the environment
print(f"Current Working Directory: {Path.cwd()}")
print("\nPython sys.path:")
for p in sys.path:
    print(f"  - {p}")
print("--- END DIAGNOSTICS ---\n")

try:
    # This is the import that is failing in the other script
    from simulations.schelling_sim import run as schelling_run
    from omegaconf import OmegaConf

    print("✅ Successfully imported 'simulations.schelling_sim.run'")

    # Mimic the setup from the main script
    config_path = "simulations/schelling_sim/config/config.yml"
    scenario_path = "simulations/schelling_sim/scenarios/default.json"

    base_config = OmegaConf.load(config_path)
    overrides = OmegaConf.create(
        {
            "scenario_path": scenario_path,
            "simulation": {"steps": 150},
        }
    )
    final_config = OmegaConf.to_container(
        OmegaConf.merge(base_config, overrides), resolve=True
    )

    print("\n🚀 Starting simulation directly...")

    schelling_run.start_simulation(
        run_id=None,
        task_id="direct_test_run",
        experiment_name="schelling-direct-test",
        config_overrides=final_config,
    )

    print("\n✅ Simulation finished successfully via direct import.")

except ImportError as e:
    print(f"\n❌ FAILED to import 'simulations.schelling_sim.run'. Error: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"\n❌ An error occurred during the direct run. Error: {e}")
    import traceback

    traceback.print_exc()
