import os
from unittest.mock import MagicMock, patch

# Subject under test
import agent_core.cognition.ai_models.openai_client as oac
import numpy as np
import pytest
from agent_core.cognition.ai_models.openai_client import (
    EmbeddingValidationError,
    get_client,
    get_embedding_from_llm,
    get_embedding_with_cache,
    get_embeddings_from_llm_batch,
    query_llm,
    validate_embedding,
)
from openai import OpenAIError


# Test Fixtures
@pytest.fixture(autouse=True)
def cleanup_openai_client():
    """Ensure the singleton client is cleared before and after each test."""
    oac._client = None
    yield
    oac._client = None


@pytest.fixture
def mock_openai(mocker):
    """Mocks the OpenAI client by patching it where it is USED."""
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    mock_client = MagicMock()

    # Mock setup for embeddings API
    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3]
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [
        mock_embedding_data,
        mock_embedding_data,
    ]  # For batch
    mock_client.embeddings.create.return_value = mock_embedding_response

    # Mock setup for chat completions API
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a test response."
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [mock_choice]
    mock_completion_response.usage = mock_usage
    mock_client.chat.completions.create.return_value = mock_completion_response

    # Patch the 'OpenAI' class within the openai_client module's namespace.
    # This ensures that when get_client() is called, it uses our mock.
    mocker.patch("agent_core.cognition.ai_models.openai_client.OpenAI", return_value=mock_client)

    return mock_client


# Test Cases

## 1. get_client() Tests


def test_get_client_success(mocker):
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    client1 = get_client()
    client2 = get_client()
    assert client1 is not None
    assert client1 is client2


def test_get_client_no_api_key_raises_error(mocker):
    # Use patch.dict to clear os.environ for the test's scope
    mocker.patch.dict(os.environ, clear=True)
    with pytest.raises(OpenAIError):
        get_client()


## 2. validate_embedding() Tests


def test_validate_embedding_success():
    assert validate_embedding(np.array([0.1, 0.2, 0.3]), 3) is True


@pytest.mark.parametrize(
    "embedding, dim, error_msg",
    [
        (None, 3, "is None"),
        ([0.1], 3, "is not a numpy array"),
        (np.array([0.1]), 3, "dimension mismatch"),
        (np.array([np.nan]), 1, "contains NaN"),
        (np.array([np.inf]), 1, "contains infinite"),
    ],
)
def test_validate_embedding_failures(embedding, dim, error_msg):
    with pytest.raises(EmbeddingValidationError, match=error_msg):
        validate_embedding(embedding, dim)


def test_validate_embedding_warns_on_zero_vector(capsys):
    validate_embedding(np.zeros(3), 3, "zero_emb")
    assert "Warning: zero_emb is a zero vector" in capsys.readouterr().out


## 3. get_embedding_from_llm() Tests


def test_get_embedding_from_llm_success(mock_openai):
    embedding = get_embedding_from_llm("test text", 3)
    assert isinstance(embedding, np.ndarray)
    mock_openai.embeddings.create.assert_called_once()


def test_get_embedding_from_llm_api_error(mock_openai, capsys):
    mock_openai.embeddings.create.side_effect = Exception("API Down")
    assert get_embedding_from_llm("test text", 3) is None
    assert "Error getting embedding from OpenAI: API Down" in capsys.readouterr().out


## 4. get_embedding_with_cache() Tests


@patch("agent_core.cognition.ai_models.openai_client.get_embedding_from_llm")
def test_get_embedding_with_cache(mock_get_embedding):
    mock_get_embedding.return_value = np.array([1.0, 2.0, 3.0])
    oac.embedding_cache.clear()
    get_embedding_with_cache("hello world", 3)
    get_embedding_with_cache("hello world", 3)  # Should hit cache
    assert mock_get_embedding.call_count == 1


## 5. get_embeddings_from_llm_batch() Tests


def test_get_embeddings_from_llm_batch_success(mock_openai):
    embeddings = get_embeddings_from_llm_batch(["text1", "text2"])
    assert len(embeddings) == 2
    mock_openai.embeddings.create.assert_called_once()


def test_get_embeddings_from_llm_batch_empty_list(mock_openai):
    assert get_embeddings_from_llm_batch([]) == []
    mock_openai.embeddings.create.assert_not_called()


def test_get_embeddings_from_llm_batch_api_error(mock_openai, capsys):
    mock_openai.embeddings.create.side_effect = Exception("Batch API Down")
    assert get_embeddings_from_llm_batch(["text1"]) is None
    assert "Error getting batch embedding from OpenAI: Batch API Down" in capsys.readouterr().out


## 6. query_llm() Tests


def test_query_llm_success(mock_openai):
    response, tokens, cost = query_llm("test prompt")
    assert response == "This is a test response."
    assert tokens == 30
    assert cost > 0


def test_query_llm_api_error(mock_openai):
    mock_openai.chat.completions.create.side_effect = Exception("Chat API Down")
    response, tokens, cost = query_llm("test prompt")
    assert "LLM reflection failed" in response
    assert tokens == 0
    assert cost == 0.0


def test_query_llm_no_usage_data(mock_openai):
    mock_openai.chat.completions.create.return_value.usage = None
    response, tokens, cost = query_llm("test prompt")
    assert response == "This is a test response."
    assert tokens == 0
    assert cost == 0.0
