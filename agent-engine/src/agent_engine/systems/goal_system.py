# src/agent_engine/systems/goal_system.py
"""
Manages an agent's goal creation, selection, and refinement.
"""

from typing import Any, Dict, List, Optional, Type, cast

import numpy as np

# Imports from agent_core
from agent_core.cognition.ai_models.openai_client import get_embedding_with_cache
from agent_core.cognition.scaffolding import CognitiveScaffold
from agent_core.core.ecs.component import (
    Component,
    EmotionComponent,
    GoalComponent,
    IdentityComponent,
    MemoryComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from sklearn.cluster import KMeans

# Imports from agent-engine
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System
from agent_engine.utils.config_utils import get_config_value
from agent_engine.utils.math_utils import safe_cosine_similarity


class GoalSystem(System):
    """
    Manages an agent's goal creation, selection, and refinement based on its
    experiences and reflections. This system is world-agnostic.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # This system is purely event-driven

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Dict[str, Any],
        cognitive_scaffold: "CognitiveScaffold",
    ) -> None:
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if not event_bus:
            raise ValueError("EventBus not initialized in SimulationState")
        self.event_bus: EventBus = event_bus

        self.event_bus.subscribe("update_goals_event", self.on_update_goals_event)

    def on_update_goals_event(self, event_data: Dict[str, Any]) -> None:
        """
        Triggered after a reflection, this orchestrates the goal update cycle.
        """
        entity_id = event_data["entity_id"]
        narrative = event_data["narrative"]
        current_tick = event_data["current_tick"]

        components = self.simulation_state.entities.get(entity_id, {})
        required_types = [
            GoalComponent,
            MemoryComponent,
            IdentityComponent,
            EmotionComponent,
        ]
        if not all(comp_type in components for comp_type in required_types):
            print(f"GoalSystem: Entity {entity_id} missing components. Skipping.")
            return

        self._invent_and_refine_goals(entity_id, components, current_tick)

        best_goal = self._select_best_goal(entity_id, components, narrative)
        goal_comp = cast(GoalComponent, components.get(GoalComponent))
        if best_goal and goal_comp:
            goal_comp.current_symbolic_goal = best_goal

    def _invent_and_refine_goals(self, entity_id: str, components: Dict[Type[Component], Component], tick: int) -> None:
        """Invent new goals by clustering successful actions and refine existing ones."""
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        goal_comp = cast(GoalComponent, components.get(GoalComponent))

        successful_actions = [m for m in mem_comp.episodic_memory if m.get("outcome", 0) > 0.1]
        if len(successful_actions) < self.config.get("goal_invention_min_successes", 5):
            return

        summaries = [
            f"Action '{m['action'].get('action_type', 'unknown')}' led to "
            f"outcome '{m['outcome_details'].get('status', 'unknown')}'"
            for m in successful_actions
        ]

        emb_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_cfg = self.config.get("llm", {})
        embeddings = np.array(
            [e for s in summaries if (e := get_embedding_with_cache(s, emb_dim, llm_cfg)) is not None]
        )

        if len(embeddings) < 3:
            return

        num_clusters = min(3, max(1, len(embeddings) // 5))
        kmeans = KMeans(n_clusters=num_clusters, random_state=tick, n_init="auto").fit(embeddings)

        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if not cluster_indices.size:
                continue

            sample_summaries = "; ".join(
                np.random.choice(
                    np.array(summaries)[cluster_indices],
                    size=min(3, len(cluster_indices)),
                    replace=False,
                )
            )
            prompt = f"""
                The following actions were successful:
                {sample_summaries}.
                What is a concise, 2-3 word,
                high-level goal that describes this pattern of success?
                (e.g., 'Assert Dominance', 'Secure Territory', 'Forge Alliances').
            """

            try:
                goal_name = (
                    self.cognitive_scaffold.query(entity_id, "goal_invention", prompt, tick)
                    .strip()
                    .lower()
                    .replace(".", "")
                )
                if goal_name and len(goal_name) > 3:
                    self._add_or_update_goal_in_component(goal_comp, entity_id, goal_name, len(cluster_indices), tick)
            except Exception as e:
                print(f"Error inventing goal for {entity_id}: {e}")

    def _add_or_update_goal_in_component(
        self,
        goal_comp: GoalComponent,
        entity_id: str,
        name: str,
        success_count: int,
        tick: int,
    ) -> None:
        """Adds a new goal or updates the success history of an existing one."""
        emb_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_cfg = self.config.get("llm", {})

        if name in goal_comp.symbolic_goals_data:
            goal_comp.symbolic_goals_data[name]["success_history"].extend([1.0] * success_count)
        else:
            embedding = get_embedding_with_cache(name, emb_dim, llm_cfg)
            if embedding is not None:
                goal_comp.symbolic_goals_data[name] = {
                    "embedding": embedding,
                    "success_history": [1.0] * success_count,
                    "last_updated_tick": tick,
                }
                print(f"AGENT {entity_id} invented new goal: '{name}'")

    def _select_best_goal(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        narrative: str,
    ) -> Optional[str]:
        """Selects the best symbolic goal based on the agent's current context."""
        goal_comp = cast(GoalComponent, components.get(GoalComponent))
        identity_comp = cast(IdentityComponent, components.get(IdentityComponent))
        emotion_comp = cast(EmotionComponent, components.get(EmotionComponent))

        if not goal_comp.symbolic_goals_data:
            return None

        emb_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        context = f"""
            My situation: {narrative}.
            My traits: {identity_comp.salient_traits_cache}.
            My emotion: {emotion_comp.current_emotion_category}.
        """

        ctx_emb = get_embedding_with_cache(context, emb_dim, self.config.get("llm", {}))
        if ctx_emb is None:
            # Assign to a locally-typed variable before returning
            # to resolve the mypy inference issue.
            current_goal: Optional[str] = goal_comp.current_symbolic_goal
            return current_goal

        # Explicitly annotate the types for 'best_goal' and 'max_score'.
        # This prevents mypy from getting confused and inferring 'Any'.
        best_goal: Optional[str] = None
        max_score: float = -float("inf")

        for name, data in goal_comp.symbolic_goals_data.items():
            score = self._score_goal(data, ctx_emb, identity_comp.embedding)
            if score > max_score:
                max_score, best_goal = score, name

        return best_goal

    def _score_goal(self, data: Dict[str, Any], ctx_emb: np.ndarray, id_emb: np.ndarray) -> float:
        """Calculates a score for a single goal based on context and identity alignment."""
        goal_emb = data.get("embedding")
        if goal_emb is None:
            return -float("inf")

        ctx_sim = safe_cosine_similarity(ctx_emb, goal_emb)
        id_sim = safe_cosine_similarity(id_emb, goal_emb)
        success_rate = np.mean(data.get("success_history", [0.5]))

        # Weights determine the importance of context vs. past success vs. identity
        # Cast to float to satisfy mypy, as numpy ops can return np.float64
        return float((ctx_sim * 2.0) + (success_rate * 1.5) + (id_sim * 1.0))

    async def update(self, current_tick: int) -> None:
        """This system is purely event-driven and has no per-tick logic."""
        pass
