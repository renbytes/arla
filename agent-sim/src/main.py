# src/main.py
"""
Example usage with Hydra:
    # Run with default settings, providing the required scenario path
    python src/main.py scenario_path=simulation/scenarios/default.json

    # Override parameters from the command line
    python src/main.py scenario_path=simulation/scenarios/cognitive_ablation.json \
    simulation.steps=50 agent.start_health=1000
"""

import os
from typing import Any, Dict  # Import Dict and Any for type hinting

import hydra  # type: ignore
from config.schemas import AppConfig
from omegaconf import DictConfig, OmegaConf  # type: ignore
from pydantic import ValidationError  # type: ignore

from src.agents.actions.base_action import Action
from src.core.simulation.engine import SimulationManager


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config: DictConfig) -> None:
    """
    Hydra-powered main function.
    The 'config' object is automatically populated by Hydra from the YAML files.
    """
    try:
        # Pydantic will parse the DictConfig and raise a detailed error if anything is wrong
        config_dict: Any = OmegaConf.to_container(config, resolve=True)

        # FIX: Add a check to ensure config_dict is a mapping before unpacking.
        # This satisfies mypy's strict type checking.
        if not isinstance(config_dict, Dict):
            print("--- CONFIGURATION ERROR ---")
            print(f"Expected configuration to be a dictionary, but got {type(config_dict)}")
            return

        validated_config = AppConfig(**config_dict)
        print("--- Configuration successfully validated ---")
    except ValidationError as e:
        print("--- CONFIGURATION ERROR ---")
        print(e)
        return  # Stop execution if config is invalid

    print("--- Configuration ---")
    print(OmegaConf.to_yaml(config))
    print("---------------------")

    # The logic for running multiple times can be handled by Hydra's multirun
    # but for a simple conversion, we can keep the loop.
    # Note: Hydra overrides will be the same for each run in this loop.
    # For true multi-runs, you'd use Hydra's command-line flags.

    # Extract config values needed here
    # Use OmegaConf.to_container to resolve any references and convert to a plain dict
    config_dict = validated_config.model_dump()
    log_dir = config_dict.get("log_directory", "logs")
    db_dir = config_dict.get("database_directory", "data")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    # Scenario path can now be passed as an override, e.g., `python main.py scenario_path=path/to/my.json`
    # We add a default value of None to the config for it.
    scenario_path = config_dict.get("scenario_path")
    if not scenario_path or not os.path.exists(scenario_path):
        print(f"ERROR: Scenario path not provided or not found: {scenario_path}")
        print(
            """Please provide it on the command line, e.g.,
            python src/main.py scenario_path=simulation/scenarios/default.json"""
        )
        return

    print("\n--- Starting simulation ---")
    Action.initialize_action_registry()

    # The SimulationManager now expects the Hydra config object `config`
    manager = SimulationManager(config=config, scenario_path=scenario_path)
    manager.run()


if __name__ == "__main__":
    main()
