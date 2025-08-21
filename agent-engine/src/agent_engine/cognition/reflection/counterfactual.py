# src/agent_engine/cognition/reflection/counterfactual.py

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

from agent_core.core.ecs.component import MemoryComponent
from agent_engine.cognition.reflection.episode import Episode
from agent_engine.simulation.simulation_state import SimulationState


@dataclass
class CounterfactualEpisode:
    """
    Represents a "what if" scenario based on a real past event, now generated
    using a formal causal model.
    """

    original_episode_theme: str
    original_action: Dict[str, Any]
    counterfactual_action: str
    predicted_outcome: str
    confidence: float


def generate_counterfactual(
    episode: Episode,
    simulation_state: SimulationState,
    agent_id: str,
    alternative_action: str,
) -> Optional[CounterfactualEpisode]:
    """
    Takes a past episode and generates a formal counterfactual "what if"
    scenario by querying the agent's CausalModel using a precise event_id.
    """
    # FIX: Cast the generic component to the specific MemoryComponent type
    mem_comp = cast(
        MemoryComponent, simulation_state.get_component(agent_id, MemoryComponent)
    )

    if (
        not episode.events
        or not mem_comp
        or not hasattr(mem_comp, "causal_model")
        or not mem_comp.causal_model
    ):
        return None

    key_event = max(episode.events, key=lambda e: abs(e.get("reward", 0.0)))
    target_event_id = key_event.get("event_id")
    if not target_event_id:
        return None

    factual_instance = next(
        (
            record
            for record in mem_comp.causal_data
            if record.get("event_id") == target_event_id
        ),
        None,
    )

    if not factual_instance:
        return None

    try:
        counterfactual_estimate = mem_comp.causal_model.whatif(
            factual_instance, treatment_value=alternative_action, outcome_name="outcome"
        )

        predicted_reward = counterfactual_estimate.value
        predicted_outcome_str = (
            f"The reward would have been approximately {predicted_reward:.2f}."
        )

        return CounterfactualEpisode(
            original_episode_theme=episode.theme,
            original_action=key_event.get("action", {}),
            counterfactual_action=alternative_action,
            predicted_outcome=predicted_outcome_str,
            confidence=0.9,
        )

    except Exception as e:
        print(f"Error generating formal counterfactual for {agent_id}: {e}")
        return None
