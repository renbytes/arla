# src/agent_engine/utils/math_utils.py

import math
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray


def safe_divide(
    numerator: float, denominator: float, default: float = 0.0, epsilon: float = 1e-12
) -> float:
    """Safely divide with stricter zero detection."""
    if abs(denominator) < epsilon or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if math.isfinite(result) else default


def safe_cosine_similarity(
    vec1: NDArray[Any], vec2: NDArray[Any], epsilon: float = 1e-8
) -> float:
    """Safely compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < epsilon or norm2 < epsilon:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def safe_normalize_vector(
    vector: Optional[NDArray[np.float32]], epsilon: float = 1e-8
) -> Optional[NDArray[np.float32]]:
    """Safely normalize a vector to unit length."""
    if vector is None:
        return None
    norm = np.linalg.norm(vector)
    if norm < epsilon:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm
