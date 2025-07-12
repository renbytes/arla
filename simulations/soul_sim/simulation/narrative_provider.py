from typing import Any, Dict, Optional, Type, cast

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.cognition.narrative_context_provider_interface import (
    NarrativeContextProviderInterface,
)
from agent_core.core.ecs.component import (
    Component,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
    SocialMemoryComponent,
)

from simulations.soul_sim.components import HealthComponent, InventoryComponent


class SoulSimNarrativeContextProvider(NarrativeContextProviderInterface):
    """Constructs the world-specific context dictionary for an agent's reflection."""

    def get_narrative_context(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        simulation_state: Any,
        current_tick: int,
    ) -> Dict[str, Any]:
        """Assembles a rich, detailed narrative context for an agent's reflection."""

        # 1. Get all necessary components safely
        health_comp = cast(HealthComponent, components.get(HealthComponent))
        inventory_comp = cast(InventoryComponent, components.get(InventoryComponent))
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        goal_comp = cast(GoalComponent, components.get(GoalComponent))
        id_comp = cast(IdentityComponent, components.get(IdentityComponent))
        social_mem_comp = cast(SocialMemoryComponent, components.get(SocialMemoryComponent))

        # --- Part A: Build the Narrative String ---

        # Section 1: Current status including vitals, goal, and identity
        status_summary = f"""I am agent {entity_id} at tick {current_tick}.
            My health is at {health_comp.normalized:.0%}
                and I possess {inventory_comp.current_resources:.1f} resources.
            My current objective is to '{goal_comp.current_symbolic_goal or "survive"}'."""
        if id_comp and id_comp.salient_traits_cache:
            status_summary += f"""I currently see myself as:
                                {", ".join(id_comp.salient_traits_cache.keys())}."""

        # Section 2: Rich summary of recent experiences, including outcomes and rewards
        event_details = []
        # Use the episodic memory, which stores more detailed event data
        for event_data in mem_comp.episodic_memory[-5:]:
            action_outcome = event_data.get("action_outcome")
            # Ensure the event has a valid outcome to report on
            if isinstance(action_outcome, ActionOutcome):
                action_name = event_data.get("action_plan", {}).get("action_type", {}).name or "an unknown action"
                status = "succeeded" if action_outcome.success else "failed"
                reward = action_outcome.reward

                # Create a more descriptive summary of what happened
                event_details.append(f"I attempted to '{action_name}', which {status} (reward: {reward:.2f}).")

        experience_summary = "My recent experiences are: " + "; ".join(event_details)

        # Combine all parts into the final narrative for the LLM
        narrative = f"{status_summary}\n{experience_summary}"

        # --- Part B: Assemble Social Feedback (re-uses existing helper) ---
        social_feedback = self._get_social_feedback(social_mem_comp)

        # --- Part C: Return the Final Context Dictionary ---
        # The structure remains the same, but the 'narrative' is now much richer.
        return {
            "narrative": narrative,
            "llm_final_account": "",
            "social_feedback": social_feedback,
        }

    def _get_social_feedback(self, social_mem_comp: Optional[SocialMemoryComponent]) -> Dict[str, float]:
        """Helper to calculate social validation scores."""
        if not social_mem_comp or not social_mem_comp.schemas:
            return {}

        positive_interactions = 0.0
        # This is a simplified example; you can build this out with your full logic
        for schema in social_mem_comp.schemas.values():
            if hasattr(schema, "impression_valence") and schema.impression_valence > 0:
                positive_interactions += 1

        return {
            "positive_social_responses": positive_interactions / len(social_mem_comp.schemas),
            "social_approval_rating": 0.0,  # Placeholder
            "peer_recognition": 0.0,  # Placeholder
        }
