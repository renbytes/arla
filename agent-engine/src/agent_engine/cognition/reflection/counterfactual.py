# src/agent_engine/cognition/reflection/counterfactual.py

from dataclasses import dataclass
from typing import Any, Dict, List

# Imports from agent_core
from agent_core.cognition.scaffolding import CognitiveScaffold

# Imports from arla-engine
from agent_engine.cognition.reflection.episode import Episode


@dataclass
class CounterfactualEpisode:
    """
    Represents a "what if" scenario based on a real past event.
    """

    original_episode_theme: str
    original_action: Dict[str, Any]
    counterfactual_action: str
    predicted_outcome: str
    confidence: float


def generate_counterfactual(
    episode: Episode,
    cognitive_scaffold: "CognitiveScaffold",
    agent_id: str,
    current_tick: int,
) -> List[CounterfactualEpisode]:
    """
    Takes a past episode and generates a counterfactual "what if" scenario
    by querying the LLM.
    """
    if not episode.events:
        return []

    # Find the event with the most significant outcome (positive or negative)
    key_event = max(episode.events, key=lambda e: abs(e.get("reward", 0.0)))

    # Construct a prompt for the LLM
    llm_prompt = (
        f"Consider the following event from an agent's life in a simulation:\n"
        f"Event: At tick {key_event.get('tick')}, the agent performed the action "
        f"'{key_event.get('action_type')}' which resulted in the outcome "
        f"'{key_event.get('status')}' and a reward of {key_event.get('reward', 0.0):.2f}.\n\n"
        "Question: What might have happened if, instead of that action, the agent had performed a "
        "completely different action (e.g., COMMUNICATE instead of COMBAT, or REST instead of EXTRACT)?\n"
        "Provide a plausible alternative action and predict the likely outcome in one sentence.\n\n"
        "Format the response as:\n"
        "ALTERNATIVE ACTION: [Action Name]\n"
        "PREDICTED OUTCOME: [Predicted outcome sentence]"
    )

    try:
        response = cognitive_scaffold.query(
            agent_id=agent_id,
            purpose="counterfactual_generation",
            prompt=llm_prompt,
            current_tick=current_tick,
        )

        lines = response.splitlines()
        alt_action = "Unknown"
        pred_outcome = "Prediction failed."

        for line in lines:
            if "ALTERNATIVE ACTION:" in line:
                alt_action = line.split(":", 1)[1].strip()
            elif "PREDICTED OUTCOME:" in line:
                pred_outcome = line.split(":", 1)[1].strip()

        return [
            CounterfactualEpisode(
                original_episode_theme=episode.theme,
                original_action=key_event.get("action", {}),
                counterfactual_action=alt_action,
                predicted_outcome=pred_outcome,
                confidence=0.8,  # Confidence is heuristic for now
            )
        ]

    except Exception as e:
        print(f"Error generating counterfactual for {agent_id}: {e}")
        return []
