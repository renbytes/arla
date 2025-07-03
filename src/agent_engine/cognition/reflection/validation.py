# src/agent_engine/cognition/reflection/validation.py
"""
Validates LLM-generated inferences against symbolic logic and episode data.
"""

from typing import Any, Dict

# Imports from agent_core
from agent_core.cognition.ai_models.openai_client import get_embedding_with_cache
from agent_core.cognition.scaffolding import CognitiveScaffold

# Imports from agent-engine
from agent_engine.cognition.reflection.episode import Episode
from agent_engine.utils.config_utils import get_config_value
from agent_engine.utils.math_utils import safe_cosine_similarity


class RuleValidator:
    """
    Validates LLM-generated inferences against symbolic logic and episode data.
    """

    def __init__(
        self,
        episode: Episode,
        config: Dict[str, Any],
        cognitive_scaffold: CognitiveScaffold,
        agent_id: str,
        current_tick: int,
    ) -> None:
        self.episode = episode
        self.config = config
        # Create a string summary of events for semantic comparison
        self.event_texts = " ".join([str(e) for e in episode.events]).lower()
        self.cognitive_scaffold = cognitive_scaffold
        self.agent_id = agent_id
        self.current_tick = current_tick

    def check_coherence(self, inference: str) -> bool:
        """A simple check for logical coherence."""
        # This can be expanded with more sophisticated checks
        if "i felt happy" in inference.lower() and "i felt sad" in inference.lower():
            return False
        return True

    def check_factual_alignment(self, inference: str) -> float:
        """
        Checks if the inference is semantically aligned with a summary
        of the episode's events, preventing token overflow.
        """
        if not inference or not self.episode.events:
            return 0.0

        embedding_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_config = self.config.get("llm", {})

        reflection_embedding = get_embedding_with_cache(inference, embedding_dim, llm_config)

        # Summarize the raw events to create a factual context
        llm_prompt = f"Concisely summarize the key events and outcomes from the following log into a single paragraph. Be factual and do not add interpretation. Data: {self.event_texts[:10000]}"

        try:
            event_summary = self.cognitive_scaffold.query(
                agent_id=self.agent_id,
                purpose="event_summary_for_validation",
                prompt=llm_prompt,
                current_tick=self.current_tick,
            )
        except Exception as e:
            print(f"Warning: Could not summarize events for validation. Error: {e}")
            return 0.0

        context_embedding = get_embedding_with_cache(event_summary, embedding_dim, llm_config)

        if reflection_embedding is None or context_embedding is None:
            print("Warning: Could not generate embeddings for validation.")
            return 0.0

        # Calculate semantic similarity between the reflection and the factual summary
        similarity = safe_cosine_similarity(reflection_embedding, context_embedding)
        alignment_score = (similarity + 1.0) / 2.0
        return alignment_score


def calculate_confidence_score(coherence: bool, factual_alignment: float, token_uncertainty: float = 0.9) -> float:
    """Combines validation checks into a single confidence score."""
    if not coherence:
        return 0.0

    confidence = (factual_alignment * 0.7) + (token_uncertainty * 0.3)
    return confidence
