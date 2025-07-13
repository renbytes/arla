# src/agent_engine/systems/reflection_system.py
"""
Manages agent metacognition, including chunking experiences into episodes,
synthesizing narratives, and triggering identity and goal updates.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, cast

# Imports from agent_core
from agent_core.cognition.narrative_context_provider_interface import (
    NarrativeContextProviderInterface,
)
from agent_core.core.ecs.component import (
    AffectComponent,
    Component,
    EmotionComponent,
    EpisodeComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
    SocialMemoryComponent,
    TimeBudgetComponent,
    ValidationComponent,
    ValueSystemComponent,
)
from agent_core.core.ecs.event_bus import EventBus

# Imports from agent_engine
from agent_engine.cognition.reflection.episode import Episode
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


class ReflectionSystem(System):
    """
    Relies on an injected NarrativeContextProviderInterface for world-specific
    narrative generation, improving decoupling.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        TimeBudgetComponent,
        AffectComponent,
        MemoryComponent,
        EpisodeComponent,
        IdentityComponent,
        GoalComponent,
        EmotionComponent,
        SocialMemoryComponent,
        ValidationComponent,
        ValueSystemComponent,
    ]

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Any,
        cognitive_scaffold: Any,
        narrative_context_provider: NarrativeContextProviderInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if not event_bus:
            raise ValueError("EventBus cannot be None for ReflectionSystem.")
        self.event_bus: EventBus = event_bus

        self.narrative_context_provider = narrative_context_provider
        self.event_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.event_bus.subscribe("action_executed", self.on_action_executed_for_chunking)
        self.event_bus.subscribe("reflection_requested_by_action", self.on_reflection_requested)

    def on_action_executed_for_chunking(self, event_data: Dict[str, Any]) -> None:
        """Buffers events for each agent to be chunked into episodes later."""
        entity_id = event_data["entity_id"]
        if entity_id not in self.event_buffer:
            self.event_buffer[entity_id] = []
        self.event_buffer[entity_id].append(event_data)

    def on_reflection_requested(self, event_data: Dict[str, Any]) -> None:
        """Handles an explicit request for reflection from another system."""
        asyncio.create_task(
            self.update_for_entity(
                entity_id=event_data["entity_id"],
                current_tick=event_data["current_tick"],
                is_final_reflection=event_data.get("is_final_reflection", False),
            )
        )

    async def update(self, current_tick: int) -> None:
        """Periodically triggers the reflection process for active agents."""
        # Use direct attribute access on the validated Pydantic model
        reflection_interval = self.config.learning.memory.reflection_interval
        if not (current_tick > 0 and current_tick % reflection_interval == 0):
            return

        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)
        for entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
            if time_comp and time_comp.is_active:
                await self._run_reflection_cycle(entity_id, components, current_tick, is_final_reflection=False)

    async def update_for_entity(self, entity_id: str, current_tick: int, is_final_reflection: bool) -> None:
        """Allows forcing a reflection cycle for a specific entity."""
        components = self.simulation_state.entities.get(entity_id)
        if components:
            await self._run_reflection_cycle(entity_id, components, current_tick, is_final_reflection)

    async def _run_reflection_cycle(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        current_tick: int,
        is_final_reflection: bool,
    ) -> None:
        """Orchestrates the full reflection process for a single agent."""
        if not self._all_required_components_present(components):
            print(f"ReflectionSystem: Skipping for {entity_id} due to missing components.")
            return

        self._chunk_and_process_episodes(entity_id, components, current_tick)
        final_context = self._synthesize_reflection(entity_id, current_tick, components)

        if final_context:
            self._validate_and_publish_outcomes(entity_id, current_tick, final_context)

    def _all_required_components_present(self, components: Dict[Type[Component], Component]) -> bool:
        """Checks if all components required by the system are present for an entity."""
        return all(comp_type in components for comp_type in self.REQUIRED_COMPONENTS)

    def _chunk_and_process_episodes(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        current_tick: int,
    ) -> None:
        """Chunks buffered events into a new Episode."""
        episode_comp = cast(EpisodeComponent, components.get(EpisodeComponent))
        if not episode_comp:
            return

        buffered_events = self.event_buffer.get(entity_id, [])
        if buffered_events:
            new_episode = self._chunk_events_into_episode(entity_id, buffered_events, current_tick)
            if new_episode:
                episode_comp.episodes.append(new_episode)
                self.event_buffer[entity_id] = []

    def _synthesize_reflection(
        self, entity_id: str, tick: int, components: Dict[Type[Component], Component]
    ) -> Dict[str, Any]:
        """Constructs the world-specific context using the provider and queries the LLM."""
        context = self.narrative_context_provider.get_narrative_context(
            entity_id=entity_id,
            components=components,
            simulation_state=self.simulation_state,
            current_tick=tick,
        )

        narrative = context.get("narrative", "")
        if not narrative:
            return context

        llm_prompt = f"""
            Based ONLY on this context: {narrative}
            Provide a concise, first-person reflection on who I am becoming,
            what I value, and what I have learned.
        """

        final_account = self.cognitive_scaffold.query(
            agent_id=entity_id,
            purpose="reflection_synthesis",
            prompt=llm_prompt,
            current_tick=tick,
        )

        context["llm_final_account"] = final_account
        return context

    def _validate_and_publish_outcomes(self, entity_id: str, tick: int, context: Dict[str, Any]) -> None:
        """Validates the reflection and publishes resulting events."""
        confidence = 0.8  # Simplified for now
        final_account = context.get("llm_final_account", "")

        self.event_bus.publish(
            "reflection_validated",
            {
                "entity_id": entity_id,
                "reflection_text": final_account,
                "confidence": confidence,
                "current_tick": tick,
            },
        )
        self.event_bus.publish(
            "update_goals_event",
            {"entity_id": entity_id, "narrative": final_account, "current_tick": tick},
        )
        self.event_bus.publish(
            "reflection_completed",
            {"tick": tick, "entity_id": entity_id, "context": context},
        )

    def _chunk_events_into_episode(self, entity_id: str, events: List[Dict[str, Any]], tick: int) -> Optional[Episode]:
        """Chunks buffered events into a single narrative Episode."""
        if not events:
            return None

        start_tick = events[0]["current_tick"]
        event_summaries = []
        for e in events:
            action_plan = e.get("action_plan")
            if action_plan and hasattr(action_plan, "action_type") and hasattr(action_plan.action_type, "name"):
                event_summaries.append(f"Tick {e['current_tick']}: action {action_plan.action_type.name}")

        if not event_summaries:
            return None

        llm_prompt = f"""
            Concisely summarize the theme of these events in 2-3 words
            (e.g., 'Territorial Dispute', 'Successful Hunt', 'Failed Exploration'):
            {"; ".join(event_summaries)}.
        """

        theme_raw = self.cognitive_scaffold.query(
            agent_id=entity_id,
            purpose="episode_theming",
            prompt=llm_prompt,
            current_tick=tick,
        )
        theme = theme_raw.strip().replace('"', "") if theme_raw else "unknown_theme"

        processed_events = [outcome.details for e in events if (outcome := e.get("action_outcome")) is not None]
        return Episode(start_tick=start_tick, end_tick=tick, theme=theme, events=processed_events)
