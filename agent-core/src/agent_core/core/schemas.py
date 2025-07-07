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


class RelationalSchema:
    """A class representing a subjective view of another agent."""

    pass
