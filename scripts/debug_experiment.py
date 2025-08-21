import sys
from pathlib import Path

from agent_sim.infrastructure.tasks.simulation_tasks import run_experiment_task
from dotenv import load_dotenv
from omegaconf import OmegaConf

# This script must be run from the project's root directory.
# Add the project root to the path to ensure all imports work correctly.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("--- Running Experiment Task Directly for Debugging ---")

# 1. Load the .env file to get environment variables like MLFLOW_TRACKING_URI
env_path = project_root / ".env"
if env_path.exists():
    print(f"✅ Loading environment variables from: {env_path}")
    load_dotenv(dotenv_path=env_path)
else:
    print(f"❌ CRITICAL: .env file not found at {env_path}. The task will fail.")
    sys.exit(1)

# 2. Load the experiment definition file
experiment_file = (
    project_root / "simulations/schelling_sim/experiments/schelling_study.yml"
)
print(f"✅ Loading experiment definition from: {experiment_file}")
exp_def = OmegaConf.load(experiment_file)

# 3. Load the base config and merge overrides for one variation
base_config_path = project_root / exp_def.base_config_path
base_config = OmegaConf.load(base_config_path)
variation = exp_def.variations[0]  # We only need to test one variation
final_config = OmegaConf.merge(base_config, variation.get("overrides", {}))
config_dict = OmegaConf.to_container(final_config, resolve=True)

# 4. Construct the arguments for the task
task_kwargs = {
    "scenario_paths": list(exp_def.scenarios),
    "runs_per_scenario": exp_def.runs_per_scenario,
    "base_config": config_dict,
    "simulation_package": exp_def.simulation_package,
    "experiment_name": f"{exp_def.experiment_name} - {variation.name}",
}

print("\n🚀 Calling the experiment task directly...")

try:
    # 5. Run the task in the current process instead of sending to a worker
    # We use .apply() which is a synchronous, direct call.
    result = run_experiment_task.apply(kwargs=task_kwargs)
    print("\n✅ Task completed successfully!")
    print(f"📄 Result: {result.get()}")
    print("\nCheck your MLflow UI and database now.")

except Exception:
    print("\n❌ An error occurred while running the task directly:")
    # This will give us the detailed traceback we need.
    import traceback

    traceback.print_exc()
