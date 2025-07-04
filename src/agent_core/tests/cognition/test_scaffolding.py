# src/agent_core/tests/cognition/test_scaffolding.py
import pytest

# Subject under test
from agent_core.cognition.scaffolding import CognitiveScaffold

# --- Test Fixtures ---


@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for CognitiveScaffold."""
    # Mock the imported query_llm function
    mock_query_llm = mocker.patch(
        "agent_core.cognition.scaffolding.query_llm",
        return_value=("LLM Response", 100, 0.001),  # (response_text, tokens, cost)
    )

    # Mock the globally instantiated db_logger and async_runner
    mock_db_logger = mocker.patch("agent_core.cognition.scaffolding.db_logger_instance")
    mock_async_runner = mocker.patch("agent_core.cognition.scaffolding.async_runner_instance")

    # Return all mocks in a dictionary for easy access in tests
    return {
        "query_llm": mock_query_llm,
        "db_logger": mock_db_logger,
        "async_runner": mock_async_runner,
    }


@pytest.fixture
def scaffold(mock_dependencies):
    """Provides a CognitiveScaffold instance with mocked dependencies."""
    # The config can be simple for this test
    config = {"llm": {"temperature": 0.5}}
    scaffold_instance = CognitiveScaffold(simulation_id="sim_123", config=config)
    return scaffold_instance


# --- Test Cases ---


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
    response = scaffold.query(agent_id=agent_id, purpose=purpose, prompt=prompt, current_tick=current_tick)

    # Assert
    # 1. Check that the response is the text part of the LLM output
    assert response == "LLM Response"

    # 2. Check that the underlying LLM function was called correctly
    mock_dependencies["query_llm"].assert_called_once_with(prompt, llm_config={"temperature": 0.5})

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

    # 4. Check that the async runner was used to call the logger
    # This verifies that the logging call is correctly wrapped for async execution.
    log_coroutine = mock_dependencies["db_logger"].log_scaffold_interaction.return_value
    mock_dependencies["async_runner"].run_async.assert_called_once_with(log_coroutine)
