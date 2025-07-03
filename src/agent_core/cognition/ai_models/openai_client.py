# src/cognition/ai_models/openai_client.py

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

# FIX: Change the client to be a lazily-initialized global variable.
# This prevents the client from being created at import time, which solves the test collection error.
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """
    Initializes and returns the OpenAI client instance on demand.
    This prevents test collection from failing when the API key is not set.
    """
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIError(
                "The OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# The rest of the functions in this file are modified to call get_client()

embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}


def get_embedding_with_cache(
    text: str, embedding_dim: int, llm_config: Optional[Dict[str, Any]] = None
) -> Optional[np.ndarray]:
    """
    Gets an embedding for a single text string, using a cache to avoid repeat API calls.
    """
    config = llm_config if llm_config else {}
    model_name = config.get("embedding_model", "text-embedding-ada-002")
    cache_key = (text, model_name)

    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    embedding = get_embedding_from_llm(text, embedding_dim, llm_config)

    if embedding is not None:
        embedding_cache[cache_key] = embedding

    return embedding


class EmbeddingValidationError(Exception):
    """Raised when embedding validation fails."""

    pass


def validate_embedding(embedding: Any, expected_dim: int, name: str = "embedding") -> bool:
    """Validate embedding dimensions and properties."""
    if embedding is None:
        raise EmbeddingValidationError(f"{name} is None")
    if not isinstance(embedding, np.ndarray):
        raise EmbeddingValidationError(f"{name} is not a numpy array")
    if embedding.shape[0] != expected_dim:
        raise EmbeddingValidationError(f"{name} dimension mismatch: got {embedding.shape[0]}, expected {expected_dim}")
    if np.any(np.isnan(embedding)):
        raise EmbeddingValidationError(f"{name} contains NaN values")
    if np.any(np.isinf(embedding)):
        raise EmbeddingValidationError(f"{name} contains infinite values")
    if np.allclose(embedding, 0):
        print(f"Warning: {name} is a zero vector, may indicate API issues")
    return True


def get_embedding_from_llm(
    text: str, expected_embedding_dim: int, llm_config: Optional[Dict[str, Any]] = None
) -> Optional[np.ndarray]:
    """Gets an embedding with proper validation."""
    client = get_client()  # FIX: Get client via the new function
    config = llm_config if llm_config else {}
    model_name = config.get("embedding_model", "text-embedding-ada-002")

    try:
        response = client.embeddings.create(input=text, model=model_name)
        embedding = np.array(response.data[0].embedding).astype(np.float32)
        validate_embedding(embedding, expected_embedding_dim, f"embedding for '{text[:50]}...'")
        return embedding
    except EmbeddingValidationError:
        raise
    except Exception as e:
        print(f"Error getting embedding from OpenAI: {e}")
        return None


def get_embeddings_from_llm_batch(
    texts: List[str], llm_config: Optional[Dict[str, Any]] = None
) -> Optional[List[np.ndarray]]:
    """
    Gets embeddings for a list of texts in a single, efficient API call.
    """
    if not texts:
        return []

    client = get_client()  # FIX: Get client via the new function
    config = llm_config if llm_config else {}
    model_name = config.get("embedding_model", "text-embedding-ada-002")

    try:
        response = client.embeddings.create(input=texts, model=model_name)
        return [np.array(data.embedding).astype(np.float32) for data in response.data]
    except Exception as e:
        print(f"Error getting batch embedding from OpenAI: {e}")
        return None


def query_llm(prompt_text: str, llm_config: Optional[Dict[str, Any]] = None) -> tuple[str, int, float]:
    """
    Queries the LLM, returning the response, token usage, and estimated cost.
    """
    client = get_client()  # FIX: Get client via the new function
    config = llm_config if llm_config else {}
    model_name = config.get("completion_model", "gpt-4.1-nano")
    temp = config.get("temperature", 0.1)
    max_tok = config.get("max_tokens", 700)
    prompt_prefix = config.get("reflection_prompt_prefix", "In 300 words or less, ")

    pricing = {
        "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
        "gpt-4.1-nano": {"prompt": 0.10 / 1_000_000, "completion": 0.40 / 1_000_000},
    }
    model_pricing = pricing.get(model_name, pricing["gpt-4o-mini"])

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_prefix + prompt_text}],
            temperature=temp,
            max_tokens=max_tok,
        )

        response_text = ""
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            response_text = response.choices[0].message.content.strip()

        usage = response.usage
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            cost = (prompt_tokens * model_pricing["prompt"]) + (completion_tokens * model_pricing["completion"])
            return response_text, total_tokens, cost
        else:
            return response_text, 0, 0.0

    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return f"LLM reflection failed due to an API error. Error: {e}", 0, 0.0
