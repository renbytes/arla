# src/agent_core/tests/cognition/test_scaffolding.py
from unittest.mock import MagicMock

import pytest

# Subject under test
from agent_core.cognition.scaffolding import CognitiveScaffold

# Test Fixtures


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for CognitiveScaffold."""
    # Mock the imported query_llm function
    mock_query_llm = mocker.patch(
        "agent_core.cognition.scaffolding.query_llm",
        return_value=("LLM Response", 100, 0.001),
    )

    # Patch asyncio.create_task where it is used, and remove the unused async_runner mock
    mock_create_task = mocker.patch(
        "agent_core.cognition.scaffolding.asyncio.create_task"
    )

    # We now mock the db_logger on the scaffold directly in the scaffold fixture
    # This is a cleaner approach than mocking a global instance.
    mock_db_logger = MagicMock()

    # Configure the mock to return an awaitable coroutine
    async def dummy_coro():
        pass

    mock_db_logger.log_scaffold_interaction.return_value = dummy_coro()

    return {
        "query_llm": mock_query_llm,
        "db_logger": mock_db_logger,  # This is now a local mock
        "create_task": mock_create_task,
    }


@pytest.fixture
def scaffold(mock_dependencies):
    """Provides a CognitiveScaffold instance with mocked dependencies."""
    # Create a mock object that mimics the Pydantic config structure.
    mock_config = MagicMock()
    # Set the .llm attribute that the scaffold's query() method expects.
    mock_config.llm = {"temperature": 0.5}

    # Pass the mock config to the constructor.
    scaffold_instance = CognitiveScaffold(
        simulation_id="sim_123",
        config=mock_config,
        db_logger=mock_dependencies["db_logger"],
    )
    return scaffold_instance


# Test Cases


def test_scaffold_query_calls_llm_and_logs_correctly(scaffold, mock_dependencies):
    """
    Tests that the query method calls the LLM, logs the interaction,
    and returns the correct response text.
    """
    # Arrange
    agent_id = "agent_x"
    purpose = "test_purpose"
    prompt = "This is a test prompt."
    current_tick = 50

    # Act
    response = scaffold.query(
        agent_id=agent_id, purpose=purpose, prompt=prompt, current_tick=current_tick
    )

    # Assert
    # 1. Check that the response is the text part of the LLM output
    assert response == "LLM Response"

    # 2. Check that the underlying LLM function was called correctly
    mock_dependencies["query_llm"].assert_called_once_with(
        prompt, llm_config={"temperature": 0.5}
    )

    # 3. Check that the database logger was called with the correct parameters
    mock_dependencies["db_logger"].log_scaffold_interaction.assert_called_once_with(
        simulation_id="sim_123",
        tick=current_tick,
        agent_id=agent_id,
        purpose=purpose,
        prompt=prompt,
        llm_response="LLM Response",
        tokens_used=100,
        cost_usd=0.001,
    )

    # 4. Check that asyncio.create_task was used to call the logger
    log_coroutine = mock_dependencies["db_logger"].log_scaffold_interaction.return_value
    mock_dependencies["create_task"].assert_called_once_with(log_coroutine)
