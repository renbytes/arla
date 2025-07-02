# src/agent_core/cognition/ai_models/openai_client.py
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Initializes and returns the OpenAI client instance on demand."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Set a dummy key if running in a test environment without a real key
            if os.getenv("PYTEST_CURRENT_TEST"):
                api_key = "test_key_not_used"
            else:
                raise OpenAIError("The OPENAI_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=api_key)
    return _client


embedding_cache: Dict[Tuple[str, str], np.ndarray] = {}


def query_llm(prompt_text: str, llm_config: Optional[Dict[str, Any]] = None) -> tuple[str, int, float]:
    """Queries the LLM, returning the response, token usage, and estimated cost."""
    # In a real scenario, this would call the OpenAI API.
    # For now, it returns a mock response.
    return f"Mock response for: {prompt_text[:50]}...", 10, 0.0001
