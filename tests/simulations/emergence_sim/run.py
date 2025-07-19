# FILE: tests/simulation/test_run.py

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

# The function we are testing
from simulations.emergence_sim.run import setup_and_run

# Test Fixtures


@pytest.fixture
def minimal_valid_config_dict():
    """
    Provides a minimal, valid configuration as a dictionary.
    This avoids file I/O and isolates the test to the logic in run.py.
    """
    return {
        "simulation": {
            "log_directory": "test_logs",
            "random_seed": 42,
            "enable_social_credit": True,
            "enable_symbol_negotiation": True,
            "enable_narrative_consensus": True,
            "enable_ritualization": True,
            "enable_normative_abstraction": True,
        },
        "agent": {
            "initial_agent_count": 5,
            "foundational": {
                "vitals": {
                    "initial_time_budget": 100,
                    "initial_health": 100,
                    "initial_resources": 10,
                },
                "attributes": {"initial_attack_power": 1, "initial_speed": 1},
            },
        },
        "environment": {"grid_world_size": [10, 10], "num_objects": 5},
        "action_modules": ["simulations.emergence_sim.actions"],
        "scenario_path": "fake/path/to/scenario.json",
        "max_ticks": 100,
    }


# Unit Tests


@pytest.mark.asyncio
async def test_setup_and_run_initialization_success(mocker, minimal_valid_config_dict):
    """
    Tests that setup_and_run can fully initialize all objects and systems
    without raising an exception, and then attempts to start the simulation.
    This is the core test for dependency injection and initialization.
    """
    # 1. ARRANGE: Mock all external dependencies
    # Mock file loading for configs
    mocker.patch("omegaconf.OmegaConf.load", return_value=minimal_valid_config_dict)

    # Mock infrastructure
    mocker.patch("simulations.emergence_sim.run.AsyncDatabaseManager")
    mocker.patch("simulations.emergence_sim.run.DatabaseEmitter")

    # Mock the scenario loader's file I/O
    mocker.patch("simulations.emergence_sim.simulation.scenario_loader.EmergenceScenarioLoader.load")

    # Mock the final `run` call - this is the key to the test
    mock_manager_run = mocker.patch(
        "agent_engine.simulation.engine.SimulationManager.run",
        new_callable=AsyncMock,  # Use AsyncMock for async functions
    )

    # 2. ACT: Call the setup function
    await setup_and_run(
        run_id="test_run_id",
        task_id="test_task_id",
        experiment_id="test_exp_id",
        config_overrides={},  # Overrides are merged into the mocked base config
    )

    # 3. ASSERT: Verify that the final step of the setup was reached
    # If this is called, it means the entire stack was created successfully.
    mock_manager_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_and_run_handles_config_validation_error(mocker):
    """
    Tests that setup_and_run correctly catches a Pydantic ValidationError
    and raises it, preventing the simulation from starting with a bad config.
    """
    # 1. ARRANGE: Mock the config loader to return invalid data
    invalid_config = {"simulation": {"log_directory": "bad"}}  # Missing many required fields
    mocker.patch("omegaconf.OmegaConf.load", return_value=invalid_config)

    # 2. ACT & ASSERT: Expect a ValidationError to be raised by the function
    with pytest.raises(ValidationError):
        await setup_and_run(
            run_id="test_run_id",
            task_id="test_task_id",
            experiment_id="test_exp_id",
            config_overrides={},
        )
