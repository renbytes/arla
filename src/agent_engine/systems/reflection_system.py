# src/systems/reflection_system.py

from typing import Any, Dict, List, Optional, Tuple, Type, cast

import numpy as np
from sqlalchemy import select

from src.cognition.reflection.counterfactual import generate_counterfactual
from src.cognition.reflection.episode import Episode
from src.cognition.reflection.narrative_engine import construct_narrative_context
from src.cognition.reflection.validation import RuleValidator, calculate_confidence_score
from src.core.ecs.abstractions import CognitiveComponent
from src.core.ecs.component import (
    AffectComponent,
    Component,
    EmotionComponent,
    EnvironmentObservationComponent,
    EpisodeComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
    PositionComponent,
    SocialMemoryComponent,
    TimeBudgetComponent,
    ValidationComponent,
    ValueSystemComponent,
)
from src.core.ecs.event_bus import EventBus
from src.core.ecs.system import System
from src.data.async_runner import async_runner
from src.data.models import AgentState, async_session_maker


class ReflectionSystem(System):
    """
    Manages agent metacognition. Refactored to use dependency injection.
    """

    REQUIRED_COMPONENTS: List[Type[CognitiveComponent]] = [TimeBudgetComponent, AffectComponent]

    def __init__(self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus: EventBus = simulation_state.event_bus
        self.event_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.event_bus.subscribe("action_executed", self.on_action_executed_for_chunking)
        self.event_bus.subscribe("reflection_requested_by_action", self.on_reflection_requested)

    def on_action_executed_for_chunking(self, event_data: Dict[str, Any]):
        """Buffers events for each agent to be chunked into episodes later."""
        entity_id = event_data["entity_id"]
        if entity_id not in self.event_buffer:
            self.event_buffer[entity_id] = []
        self.event_buffer[entity_id].append(event_data)

    def on_reflection_requested(self, event_data: Dict[str, Any]):
        """
        Handles an explicit request for reflection from another system (e.g., ReflectionActionSystem).
        """
        self.update_for_entity(
            entity_id=event_data["entity_id"],
            current_tick=event_data["current_tick"],
            is_final_reflection=event_data.get("is_final_reflection", False),
        )

    def update(self, current_tick: int):
        """
        Periodically triggers the reflection process for all active agents.
        """
        simulation_config = self.config.get("simulation", {})
        learning_memory_config = self.config.get("learning", {}).get("memory", {})

        if current_tick >= simulation_config.get("steps", 400) - 1:
            return

        reflection_interval = learning_memory_config.get("reflection_interval", 50)
        is_periodic_tick = current_tick > 0 and current_tick % reflection_interval == 0

        # Use type: ignore for the known list variance issue
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)  # type: ignore[arg-type]

        for entity_id, components in target_entities.items():
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
            if not time_comp or not time_comp.is_active:
                continue

            should_reflect = False
            if is_periodic_tick:
                should_reflect = True
            else:
                affect_comp = cast(AffectComponent, components.get(AffectComponent))
                if affect_comp and len(affect_comp.dissonance_history) > 20:
                    history = np.array(list(affect_comp.dissonance_history))
                    threshold = np.mean(history) + (1.0 * np.std(history))
                    if affect_comp.cognitive_dissonance > threshold:
                        should_reflect = True
                        print(f"INFO: Dynamic reflection triggered for {entity_id} at tick {current_tick}.")

            if should_reflect:
                all_entity_components = self.simulation_state.entities.get(entity_id, {})
                self._run_reflection_cycle(entity_id, all_entity_components, current_tick, is_final_reflection=False)

    def update_for_entity(self, entity_id: str, current_tick: int, is_final_reflection: bool):
        """Allows forcing a reflection cycle for a specific entity (e.g., at simulation end)."""
        components = self.simulation_state.entities.get(entity_id)
        if components:
            self._run_reflection_cycle(entity_id, components, current_tick, is_final_reflection)

    def _run_reflection_cycle(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        current_tick: int,
        is_final_reflection: bool,
    ):
        """
        Orchestrates the full reflection process for a single agent.
        """
        component_map = self._gather_required_components(entity_id, components)
        if not component_map:
            return

        self._chunk_and_process_episodes(entity_id, component_map, current_tick, is_final_reflection)

        final_account, narrative_context = self._synthesize_reflection(entity_id, current_tick, component_map)

        if final_account:
            self._validate_and_publish_outcomes(
                entity_id, current_tick, final_account, narrative_context, component_map
            )

    def _gather_required_components(self, entity_id: str, components: Dict) -> Optional[Dict[str, Any]]:
        """Gathers all components required for reflection and returns them in a map."""
        required_types = [
            MemoryComponent,
            AffectComponent,
            IdentityComponent,
            GoalComponent,
            EpisodeComponent,
            ValidationComponent,
            ValueSystemComponent,
            PositionComponent,
            EnvironmentObservationComponent,
            EmotionComponent,
            SocialMemoryComponent,
        ]
        component_map = {comp_type.__name__: components.get(comp_type) for comp_type in required_types}
        if not all(component_map.values()):
            print(f"Warning: {entity_id} missing one or more required components for reflection. Aborting cycle.")
            return None
        return component_map

    def _chunk_and_process_episodes(
        self, entity_id: str, component_map: Dict, current_tick: int, is_final_reflection: bool
    ):
        """Chunks buffered events into a new Episode and, if not the final reflection, generates counterfactuals."""
        episode_comp = component_map["EpisodeComponent"]
        mem_comp = component_map["MemoryComponent"]

        buffered_events = self.event_buffer.get(entity_id, [])
        if buffered_events:
            new_episode = self._chunk_events_into_episode(
                entity_id, buffered_events, self.simulation_state.simulation_id
            )
            if new_episode:
                episode_comp.episodes.append(new_episode)
                self.event_buffer[entity_id] = []

        if not is_final_reflection and episode_comp.episodes:
            if counterfactuals := generate_counterfactual(
                episode=episode_comp.episodes[-1],
                cognitive_scaffold=self.cognitive_scaffold,
                agent_id=entity_id,
                current_tick=current_tick,
            ):
                mem_comp.counterfactual_memories.extend(counterfactuals)

    def _synthesize_reflection(
        self, entity_id: str, current_tick: int, component_map: Dict
    ) -> Tuple[Optional[str], str]:
        """Constructs the narrative, queries the LLM, and logs the interaction."""
        mem_comp = component_map["MemoryComponent"]
        narrative_context = construct_narrative_context(
            agent_id=entity_id,
            episodic_memory=mem_comp.episodic_memory,
            causal_graph=mem_comp.causal_graph,
            social_memory=component_map["SocialMemoryComponent"].schemas,
            value_system=component_map["ValueSystemComponent"],
            current_tick=current_tick,
            agent_pos_history=component_map["PositionComponent"].history,
            agent_state_obj=component_map["EnvironmentObservationComponent"],
            simulation_state=self.simulation_state,
            goal_comp=component_map["GoalComponent"],
        )
        llm_prompt = f"""Based ONLY on this context: {narrative_context}
            Provide a concise, first-person reflection.
            Synthesize who I am becoming, what I value, and what I have learned."""

        final_account = self.cognitive_scaffold.query(
            agent_id=entity_id, purpose="reflection_synthesis", prompt=llm_prompt, current_tick=current_tick
        )
        mem_comp.last_llm_reflection_summary = final_account
        return final_account, narrative_context

    def _validate_and_publish_outcomes(
        self, entity_id: str, current_tick: int, final_account: str, narrative_context: str, component_map: Dict
    ):
        """Validates the reflection and publishes all resulting events."""
        episode_comp = component_map["EpisodeComponent"]
        validation_comp = component_map["ValidationComponent"]
        identity_comp = component_map["IdentityComponent"]

        confidence = None

        if episode_comp.episodes:
            validator = RuleValidator(
                episode=episode_comp.episodes[-1],
                config=self.config,
                cognitive_scaffold=self.cognitive_scaffold,
                agent_id=entity_id,
                current_tick=current_tick,
            )
            confidence = calculate_confidence_score(
                validator.check_coherence(final_account), validator.check_factual_alignment(final_account)
            )
            validation_comp.reflection_confidence_scores[current_tick] = confidence
            print(f"Reflection for {entity_id} validated with confidence: {confidence:.2f}")

            if confidence > 0.7:
                self.event_bus.publish(
                    "reflection_validated",
                    {
                        "entity_id": entity_id,
                        "reflection_text": final_account,
                        "confidence": confidence,
                        "current_tick": current_tick,
                    },
                )

        self.event_bus.publish(
            "update_goals_event", {"entity_id": entity_id, "narrative": final_account, "current_tick": current_tick}
        )
        self.event_bus.publish(
            "reflection_completed",
            {
                "tick": current_tick,
                "entity_id": entity_id,
                "llm_final_account": final_account,
                "narrative_context": narrative_context,
                "confidence": confidence,
                "identity_coherence": identity_comp.multi_domain_identity.get_identity_coherence(),
                "identity_stability": identity_comp.get_identity_stability(),
            },
        )

    def _get_goal_at_tick(self, sim_id: str, agent_id: str, tick: int) -> str:
        """
        Helper function to query the database for a historical goal state.
        """

        async def _fetch_goal():
            async with async_session_maker() as session:
                result = await session.execute(
                    select(AgentState.current_goal).where(
                        AgentState.simulation_id == sim_id, AgentState.agent_id == agent_id, AgentState.tick == tick
                    )
                )
                state = result.scalar_one_or_none()
                return str(state) if state else "unknown"

        try:
            return async_runner.run_async(_fetch_goal())
        except Exception as e:
            print(f"Error in _get_goal_at_tick: {e}")
            return "unknown"

    def _chunk_events_into_episode(self, entity_id: str, events: List[Dict], sim_id: str) -> Optional[Episode]:
        """Chunks buffered events into a single narrative Episode and logs the theming interaction."""
        if not events:
            return None

        goal_comp = self.simulation_state.get_component(entity_id, GoalComponent)

        start_tick = events[0]["current_tick"]
        end_tick = events[-1]["current_tick"]
        event_summaries = "; ".join(
            [
                f"At tick {e['current_tick']}, my action {e['action_plan'].action_type.name} resulted in '{e['action_outcome'].details.get('status', 'unknown')}'"
                for e in events
            ]
        )

        llm_prompt = f"""Based on these events: {event_summaries}.
            What is a two or three word theme for this episode?
            (e.g., 'fruitless search', 'successful collaboration', 'unexpected conflict')"""

        theme = (
            self.cognitive_scaffold.query(
                agent_id=entity_id, purpose="episode_theming", prompt=llm_prompt, current_tick=end_tick
            )
            .strip()
            .replace('"', "")
        )

        goal_at_end = "unknown"
        if isinstance(goal_comp, GoalComponent):
            goal_at_end = goal_comp.current_symbolic_goal or "unknown"

        return Episode(
            start_tick=start_tick,
            end_tick=end_tick,
            theme=theme or "untitled episode",
            emotional_valence_curve=[e["action_outcome"].reward for e in events],
            events=[e["action_outcome"].details for e in events],
            goal_at_start=self._get_goal_at_tick(sim_id, entity_id, start_tick),
            goal_at_end=goal_at_end,
        )
