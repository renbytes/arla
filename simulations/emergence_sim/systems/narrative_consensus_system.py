# simulations/emergence_sim/systems/narrative_consensus_system.py
"""
This system manages the social process of storytelling and validation.
It listens
for shared narratives and allows agents to evaluate them against their own
experiences, fostering a collective memory or leading to cognitive dissonance.
"""

import asyncio
from typing import Any, Dict, List, Type, cast

from agent_core.core.ecs.component import (
    AffectComponent,
    BeliefSystemComponent,
    Component,
    MemoryComponent,
)
from agent_core.core.schemas import Belief
from agent_engine.simulation.system import System

from simulations.emergence_sim.components import PositionComponent


class NarrativeConsensusSystem(System):
    """
    Manages the social process of storytelling and validation.
    This system is event-driven. When an agent shares a narrative, this system
    identifies nearby listeners. Each listener then asynchronously evaluates the
    shared story against its own memories, using the CognitiveScaffold (LLM).
    This can lead to the adoption of new beliefs or an increase in cognitive
    dissonance if the narratives conflict.
    """

    # This system acts on listeners, so it needs their components.
    REQUIRED_COMPONENTS: List[Type[Component]] = [
        MemoryComponent,
        AffectComponent,
        BeliefSystemComponent,
        PositionComponent,
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the system and subscribes to the narrative_shared event."""
        super().__init__(*args, **kwargs)
        if not self.event_bus:
            raise ValueError("NarrativeConsensusSystem requires an EventBus.")
        # The ShareNarrativeAction publishes a "narrative_shared" event.
        self.event_bus.subscribe("narrative_shared", self.on_narrative_shared)

    def on_narrative_shared(self, event_data: Dict[str, Any]) -> None:
        """
        Event handler triggered when a narrative is shared.
        It finds nearby
        listeners and spawns asynchronous tasks to process their responses.
        """
        speaker_id = event_data.get("speaker_id")
        narrative = event_data.get("narrative")
        current_tick = event_data.get("tick")

        if not all([speaker_id, narrative, isinstance(current_tick, int)]):
            return

        speaker_pos_comp = self.simulation_state.get_component(speaker_id, PositionComponent)
        if not isinstance(speaker_pos_comp, PositionComponent):
            return

        # Find all entities within a certain radius to act as listeners
        if not self.simulation_state.environment:
            return
        nearby_entities = self.simulation_state.environment.get_entities_in_radius(speaker_pos_comp.position, radius=3)

        for listener_id, _ in nearby_entities:
            if listener_id == speaker_id:
                continue  # An agent doesn't listen to its own story

            # Launch a non-blocking task for each listener
            asyncio.create_task(self._process_listener_response(listener_id, speaker_id, narrative, current_tick))

    async def _process_listener_response(
        self, listener_id: str, speaker_id: str, shared_narrative: str, tick: int
    ) -> None:
        """
        Asynchronously processes a single listener's cognitive response to a
        shared narrative.
        """
        listener_components = self.simulation_state.entities.get(listener_id)
        if not listener_components or not all(
            comp_type in listener_components for comp_type in self.REQUIRED_COMPONENTS
        ):
            return

        mem_comp = cast(MemoryComponent, listener_components[MemoryComponent])
        affect_comp = cast(AffectComponent, listener_components[AffectComponent])
        belief_comp = cast(BeliefSystemComponent, listener_components[BeliefSystemComponent])

        # Use the agent's own last reflection as a basis for comparison
        own_experience = (
            mem_comp.last_llm_reflection_summary
            if mem_comp.last_llm_reflection_summary
            else "I have no recent experiences of note."
        )

        prompt = f"""
        I am an agent in a simulation.
        Another agent (ID: {speaker_id}) just told me this story:
        "{shared_narrative}"

        My own recent experience is:
        "{own_experience}"

        Compare these two accounts.
        Respond in one word: ALIGN, CONFLICT, or UNRELATED.
        - ALIGN: If the stories are about the same events and are compatible.
        - CONFLICT: If the stories are about the same events but contradict each other.
        - UNRELATED: If the stories are about different events.
        """

        try:
            response = (
                self.cognitive_scaffold.query(
                    agent_id=listener_id,
                    purpose="narrative_evaluation",
                    prompt=prompt,
                    current_tick=tick,
                )
                .strip()
                .upper()
            )

            if "ALIGN" in response:
                # If narratives align, adopt the shared narrative as a belief
                new_belief = Belief(
                    statement=shared_narrative,
                    confidence=0.7,  # Initial confidence from social validation
                    source_reflection_tick=tick,
                )
                belief_comp.belief_base[f"narrative_{tick}_{speaker_id}"] = new_belief
                print(f"INFO: Agent {listener_id} aligned with narrative from {speaker_id}.")

            elif "CONFLICT" in response:
                # If narratives conflict, increase cognitive dissonance
                affect_comp.cognitive_dissonance = min(1.0, affect_comp.cognitive_dissonance + 0.5)
                print(f"INFO: Agent {listener_id} experienced cognitive dissonance from narrative by {speaker_id}.")

        except Exception as e:
            print(f"ERROR: LLM query failed for narrative consensus for agent {listener_id}: {e}")

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven and has no per-tick logic."""
        pass
