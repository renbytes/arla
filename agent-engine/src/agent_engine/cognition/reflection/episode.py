# src/agent_engine/cognition/reflection/episode.py

from dataclasses import dataclass, field
from typing import Any, Dict, List

from agent_core.core.schemas import Episode as CoreEpisode


@dataclass
class Episode(CoreEpisode):
    """
    Represents a temporally and thematically coherent chunk of an agent's
    experience, used for creating narrative arcs for reflection.
    """

    start_tick: int
    end_tick: int
    theme: str  # e.g., "trust_violation", "successful_collaboration"
    emotional_valence_curve: List[float] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    goal_at_start: str = "unknown"
    goal_at_end: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Converts the episode to a dictionary for logging or serialization."""
        return {
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "theme": self.theme,
            "emotional_valence_curve": self.emotional_valence_curve,
            "goal_at_start": self.goal_at_start,
            "goal_at_end": self.goal_at_end,
            "event_count": len(self.events),
            # Do not serialize all events to avoid large logs
            "event_previews": [
                f"Tick {e.get('tick')}: {e.get('action_type')}" for e in self.events[:3]
            ],
        }
