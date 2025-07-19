# tests/utils/test_math_utils.py

import numpy as np
import pytest

# Subject under test
from agent_engine.utils.math_utils import (
    safe_cosine_similarity,
    safe_divide,
    safe_normalize_vector,
)

# Test Cases for safe_divide


@pytest.mark.parametrize(
    "numerator, denominator, default, expected",
    [
        (10, 2, 0.0, 5.0),  # Standard division
        (10, 0, 99.0, 99.0),  # Division by zero, returns default
        (10, 1e-15, 99.0, 99.0),  # Division by number smaller than epsilon
        (0, 10, 0.0, 0.0),  # Numerator is zero
        (float("inf"), 10, 0.0, 0.0),  # Numerator is infinity
        (10, float("nan"), 0.0, 0.0),  # Denominator is NaN
    ],
)
def test_safe_divide(numerator, denominator, default, expected):
    """
    Tests safe_divide with various inputs, including edge cases.
    """
    assert safe_divide(numerator, denominator, default) == pytest.approx(expected)


# Test Cases for safe_cosine_similarity


def test_safe_cosine_similarity_identical_vectors():
    """
    Tests that identical vectors have a similarity of 1.0.
    """
    vec = np.array([1, 2, 3])
    assert safe_cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_safe_cosine_similarity_opposite_vectors():
    """
    Tests that opposite vectors have a similarity of -1.0.
    """
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([-1, -2, -3])
    assert safe_cosine_similarity(vec1, vec2) == pytest.approx(-1.0)


def test_safe_cosine_similarity_orthogonal_vectors():
    """
    Tests that orthogonal vectors have a similarity of 0.0.
    """
    vec1 = np.array([1, 0])
    vec2 = np.array([0, 1])
    assert safe_cosine_similarity(vec1, vec2) == pytest.approx(0.0)


def test_safe_cosine_similarity_with_zero_vector():
    """
    Tests that similarity with a zero-length vector is 0.0.
    """
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([0, 0, 0])
    assert safe_cosine_similarity(vec1, vec2) == pytest.approx(0.0)
    assert safe_cosine_similarity(vec2, vec1) == pytest.approx(0.0)


# Test Cases for safe_normalize_vector


def test_safe_normalize_vector_standard():
    """
    Tests that a standard vector is correctly normalized to unit length.
    """
    vec = np.array([3, 4], dtype=np.float32)  # Length is 5
    normalized_vec = safe_normalize_vector(vec)
    assert normalized_vec is not None
    np.testing.assert_allclose(normalized_vec, np.array([0.6, 0.8]))
    assert np.linalg.norm(normalized_vec) == pytest.approx(1.0)


def test_safe_normalize_vector_zero_vector():
    """
    Tests that a zero-length vector is safely handled and returns a zero vector.
    """
    vec = np.array([0, 0, 0], dtype=np.float32)
    normalized_vec = safe_normalize_vector(vec)
    assert normalized_vec is not None
    np.testing.assert_array_equal(normalized_vec, np.zeros_like(vec))


def test_safe_normalize_vector_none_input():
    """
    Tests that passing None to the function returns None.
    """
    assert safe_normalize_vector(None) is None
