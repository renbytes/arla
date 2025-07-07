# src/agent_core/core/schemas.py

from dataclasses import dataclass


@dataclass
class Belief:
    """A data class representing a single belief."""

    statement: str
    confidence: float
    source_reflection_tick: int


@dataclass
class Episode:
    """A data class representing a narrative episode."""

    pass


@dataclass
class CounterfactualEpisode:
    """A data class representing a 'what if' scenario."""

    pass


@dataclass
class RelationalSchema:
    """A class representing a subjective view of another agent."""

    other_agent_id: str
    impression_valence: float = 0.0
    interaction_count: int = 0
