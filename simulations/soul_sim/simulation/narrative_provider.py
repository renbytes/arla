from typing import Any, Dict, Optional, Type, cast

from agent_core.cognition.narrative_context_provider_interface import (
    NarrativeContextProviderInterface,
)
from agent_core.core.ecs.component import (
    Component,
    GoalComponent,
    MemoryComponent,
    SocialMemoryComponent,
)


class SoulSimNarrativeContextProvider(NarrativeContextProviderInterface):
    """Constructs the world-specific context dictionary for an agent's reflection."""

    def get_narrative_context(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        simulation_state: Any,
        current_tick: int,
    ) -> Dict[str, Any]:
        """Assembles and returns a dictionary of all world-specific context."""

        # 1. Assemble the narrative string
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        goal_comp = cast(GoalComponent, components.get(GoalComponent))

        event_summary = "My recent experiences include: " + "; ".join(
            [f"action {e.get('action_plan',{}).get('action_type',{}).name}" for e in mem_comp.episodic_memory[-5:]]
        )
        goal_summary = f"My current goal is {goal_comp.current_symbolic_goal if goal_comp else 'unknown'}."
        narrative = f"I am {entity_id} at tick {current_tick}. {goal_summary}\n{event_summary}"

        # 2. Assemble the social feedback data
        social_mem_comp = cast(SocialMemoryComponent, components.get(SocialMemoryComponent))
        social_feedback = self._get_social_feedback(social_mem_comp)

        # 3. Return the complete context dictionary
        return {
            "narrative": narrative,
            "llm_final_account": "", # This will be filled in by the ReflectionSystem
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
