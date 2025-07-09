# # tests/infrastructure/tasks/test_simulation_tasks.py
# """
# Comprehensive unit tests for the Celery tasks in simulation_tasks.py.
# """
# import asyncio
# from unittest.mock import ANY, MagicMock, patch

# import pytest
# from agent_engine.simulation.engine import SimulationManager
# from omegaconf import OmegaConf
# from pydantic import ValidationError

# # Subject under test
# from src.agent_sim.infrastructure.tasks.simulation_tasks import (
#     _handle_simulation_exception,
#     _initialize_manager,
#     health_check,
#     run_experiment_task,
#     run_simulation_task,
# )

# # --- Mocks and Fixtures ---

# @pytest.fixture(autouse=True)
# def mock_current_task(monkeypatch):
#     """
#     Celery's `current_task` is only available inside a worker context.
#     This fixture patches it globally for all tests in this module
#     to prevent `AttributeError: 'NoneType' has no attribute 'request'`.
#     """
#     task = MagicMock(name="MockCeleryTask")
#     task.request.id = "mock-celery-task-id"
#     task.retry.side_effect = Exception("RetryCalled")
#     monkeypatch.setattr(
#         "src.agent_sim.infrastructure.tasks.simulation_tasks.current_task",
#         task
#     )
#     return task

# @pytest.fixture
# def minimal_config():
#     """Provides a minimal, valid config dictionary for testing."""
#     return OmegaConf.create({
#         "simulation_name": "test_sim",
#         "providers": {
#             "scenario_loader": {"_target_": "unittest.mock.MagicMock"},
#             "action_generator": {"_target_": "unittest.mock.MagicMock"},
#             "decision_selector": {"_target_": "unittest.mock.MagicMock"},
#         },
#         "environment": {
#             "_target_": "unittest.mock.MagicMock",
#             "params": {"width": 10, "height": 10},
#         },
#         "engine_systems_to_load": [],
#         "world_systems_to_load": [],
#     })

# # --- Test Health Check ---

# def test_health_check_returns_healthy_status():
#     """Tests that the health_check task returns the correct dictionary."""
#     result = health_check()
#     assert result == {"status": "healthy", "message": "Worker is operational"}

# # --- Test _handle_simulation_exception Helper ---

# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.mlflow")
# def test_handle_simulation_exception_retries_on_db_lock(mock_mlflow, mock_current_task):
#     """Tests that the handler correctly identifies a transient error and retries."""
#     exc = Exception("database is locked")
#     with pytest.raises(Exception, match="RetryCalled"):
#         _handle_simulation_exception(exc, mock_current_task, "task-id")

#     mock_mlflow.set_tag.assert_called_with("status", "failed")
#     mock_current_task.retry.assert_called_once()

# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.mlflow")
# def test_handle_simulation_exception_raises_other_errors(mock_mlflow, mock_current_task):
#     """Tests that the handler re-raises non-transient exceptions."""
#     exc = ValueError("A non-retryable error")
#     with pytest.raises(ValueError, match="A non-retryable error"):
#         _handle_simulation_exception(exc, mock_current_task, "task-id")

#     mock_mlflow.set_tag.assert_called_with("status", "failed")
#     mock_current_task.retry.assert_not_called()

# # --- Test run_simulation_task ---

# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.asyncio.run")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks._initialize_manager")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.mlflow")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.Action")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.AppConfig")
# def test_run_simulation_task_successful_path(
#     mock_app_config, mock_action, mock_mlflow, mock_init_manager, mock_asyncio_run,
#     mock_current_task, minimal_config
# ):
#     """Tests the entire successful execution path of a single simulation run."""
#     # ARRANGE
#     mock_manager = MagicMock(spec=SimulationManager)
#     mock_manager.simulation_id = "sim-id-success"
#     mock_manager.save_path = "/tmp/sim_results"
#     mock_init_manager.return_value = mock_manager

#     # ACT: Call the wrapped function with all keyword arguments.
#     result = run_simulation_task.__wrapped__(
#         self=mock_current_task,
#         scenario_path="/path/to/scenario.json",
#         config_overrides=minimal_config,
#         run_id="run-id-success",
#         experiment_name="Test-Experiment"
#     )

#     # ASSERT
#     mock_app_config.assert_called_once_with(**minimal_config)
#     mock_action.initialize_action_registry.assert_called_once()
#     mock_mlflow.set_experiment.assert_called_with("Test-Experiment")
#     mock_mlflow.start_run.assert_called_once_with(run_name="run-id-success")
#     mock_init_manager.assert_called_once()
#     mock_asyncio_run.assert_called_once_with(mock_manager.run())

#     assert result["status"] == "completed"
#     assert result["run_id"] == "run-id-success"

# def test_run_simulation_task_handles_validation_error(mock_current_task, minimal_config):
#     """Tests that the task correctly handles a Pydantic ValidationError."""
#     with patch("src.agent_sim.infrastructure.tasks.simulation_tasks.AppConfig", side_effect=ValidationError.from_exception_data("Error", [])):
#         with pytest.raises(ValidationError):
#             # Call the wrapped function with all keyword arguments.
#             run_simulation_task.__wrapped__(
#                 self=mock_current_task,
#                 scenario_path="/path/to/scenario.json",
#                 config_overrides=minimal_config,
#             )

# # --- Test run_experiment_task ---

# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.run_simulation_task")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.async_runner")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.AsyncDatabaseManager")
# @patch("src.agent_sim.infrastructure.tasks.simulation_tasks.mlflow")
# def test_run_experiment_task_submits_correct_jobs(
#     mock_mlflow, mock_db_manager, mock_async_runner, mock_run_sim_task,
#     mock_current_task, minimal_config
# ):
#     """
#     Tests that the experiment task correctly calculates and submits the right number of jobs.
#     """
#     # ARRANGE
#     scenario_paths = ["s1.json", "s2.json"]
#     runs_per_scenario = 3

#     mock_s_obj = MagicMock()
#     mock_set_obj = MagicMock()
#     mock_delay_obj = MagicMock()
#     mock_s_obj.set.return_value = mock_set_obj
#     mock_set_obj.delay.return_value = mock_delay_obj
#     mock_run_sim_task.s.return_value = mock_s_obj

#     # ACT: Call the wrapped function with all keyword arguments.
#     result = run_experiment_task.__wrapped__(
#         self=mock_current_task,
#         scenario_paths=scenario_paths,
#         runs_per_scenario=runs_per_scenario,
#         base_config=minimal_config,
#         simulation_package="soul_sim",
#         experiment_name="Multi-Run-Test"
#     )

#     # ASSERT
#     mock_db_manager.return_value.create_experiment.assert_called_once()
#     assert result["total_jobs"] == 6
#     assert mock_run_sim_task.s.call_count == 6
#     mock_s_obj.set.assert_called_with(queue="simulations")
