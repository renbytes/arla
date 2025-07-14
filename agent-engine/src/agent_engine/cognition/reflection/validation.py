# src/agent_engine/cognition/reflection/validation.py
"""
Validates LLM-generated inferences and formal causal models.
"""

from typing import Any, Optional

from agent_core.cognition.ai_models.openai_client import get_embedding_with_cache
from agent_core.cognition.scaffolding import CognitiveScaffold
from agent_engine.cognition.reflection.episode import Episode
from agent_engine.utils.math_utils import safe_cosine_similarity
from dowhy import CausalModel


class CausalModelValidator:
    """
    Validates a dowhy CausalModel using refutation methods to ensure its
    conclusions are robust.
    """

    def __init__(self, causal_model: CausalModel):
        self.model = causal_model
        self.estimand = self.model.identify_effect(proceed_when_unidentifiable=True)
        self.estimate = self.model.estimate_effect(self.estimand, method_name="backdoor.linear_regression")

    def check_robustness(self) -> dict:
        """
        Runs a suite of refutation tests on the causal model.

        Returns:
            A dictionary containing the results and confidence scores from each test.
        """
        results = {
            "random_common_cause": self._refute_with_random_common_cause(),
            "placebo_treatment": self._refute_with_placebo_treatment(),
            "data_subset": self._refute_with_data_subset(),
        }
        return results

    def _refute_with_random_common_cause(self) -> Optional[float]:
        """
        Tests how sensitive the model is to an unobserved common cause.
        A robust model's estimate should not change significantly.
        Returns a confidence score from 0.0 to 1.0.
        """
        try:
            refute = self.model.refute_estimate(self.estimand, self.estimate, method_name="add_unobserved_common_cause")
            # Confidence is high if the new estimate is close to the original
            original_value = self.estimate.value
            new_value = refute.new_effect
            confidence = 1.0 - min(1.0, abs(new_value - original_value) / (abs(original_value) + 1e-6))
            return confidence
        except Exception as e:
            print(f"Causal refutation (random common cause) failed: {e}")
            return None

    def _refute_with_placebo_treatment(self) -> Optional[float]:
        """
        Replaces the actual treatment (action) with a placebo.
        A robust model should show a near-zero effect for the placebo.
        Returns a confidence score from 0.0 to 1.0.
        """
        try:
            refute = self.model.refute_estimate(self.estimand, self.estimate, method_name="placebo_treatment_refuter")
            # Confidence is high if the placebo effect is close to zero
            confidence = 1.0 - min(1.0, abs(refute.new_effect))
            return confidence
        except Exception as e:
            print(f"Causal refutation (placebo treatment) failed: {e}")
            return None

    def _refute_with_data_subset(self) -> Optional[float]:
        """
        Retrains the model on a random subset of the data.
        A robust model should yield a similar estimate.
        Returns a confidence score from 0.0 to 1.0.
        """
        try:
            refute = self.model.refute_estimate(self.estimand, self.estimate, method_name="data_subset_refuter")
            # Confidence is high if the new estimate is close to the original
            original_value = self.estimate.value
            new_value = refute.new_effect
            confidence = 1.0 - min(1.0, abs(new_value - original_value) / (abs(original_value) + 1e-6))
            return confidence
        except Exception as e:
            print(f"Causal refutation (data subset) failed: {e}")
            return None


class RuleValidator:
    """
    Validates LLM-generated inferences against symbolic logic and episode data.
    """

    def __init__(
        self,
        episode: Episode,
        config: Any,
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

        # Use direct attribute access
        embedding_dim = self.config.agent.cognitive.embeddings.main_embedding_dim
        llm_config = self.config.llm

        reflection_embedding = get_embedding_with_cache(inference, embedding_dim, llm_config)

        # Summarize the raw events to create a factual context
        llm_prompt = f"""Concisely summarize the key events and outcomes from the
            following log into a single paragraph.
            Be factual and do not add interpretation.
            Data: {self.event_texts[:10000]}
        """

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
