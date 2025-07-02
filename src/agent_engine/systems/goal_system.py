# src/systems/goal_system.py

from typing import Any, Dict, List, Optional, Type, cast

import numpy as np
from sklearn.cluster import KMeans  # type: ignore

from src.cognition.ai_models.openai_client import get_embedding_with_cache
from src.core.ecs.abstractions import CognitiveComponent
from src.core.ecs.component import (
    AffectComponent,
    Component,
    EmotionComponent,
    GoalComponent,
    HealthComponent,
    IdentityComponent,
    InventoryComponent,
    MemoryComponent,
    TimeBudgetComponent,
)
from src.core.ecs.event_bus import EventBus
from src.core.ecs.system import System
from src.utils.config_utils import get_config_value
from src.utils.math_utils import safe_cosine_similarity


class GoalSystem(System):
    """
    Manages agent goal creation, selection, and refinement.
    """

    # purely event-driven
    REQUIRED_COMPONENTS: List[Type[CognitiveComponent]] = []

    def __init__(self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus: EventBus = simulation_state.event_bus
        self.event_bus.subscribe("update_goals_event", self.on_update_goals_event)
        self.llm_client = simulation_state.llm_client
        self.config = config

    # ------------------------------------------------------------------ #
    # event handler
    # ------------------------------------------------------------------ #
    def on_update_goals_event(self, event_data: Dict[str, Any]) -> None:
        entity_id: str = event_data["entity_id"]
        narrative: str = event_data["narrative"]
        current_tick: int = event_data["current_tick"]

        components: Dict[Type[Component], Component] = self.simulation_state.entities.get(entity_id, {})

        goal_comp = components.get(GoalComponent)
        if not isinstance(goal_comp, GoalComponent):
            return

        # verify required components exist (and use Type[Component] to appease mypy)
        required_types: List[Type[Component]] = [
            MemoryComponent,
            HealthComponent,
            TimeBudgetComponent,
            InventoryComponent,
            AffectComponent,
        ]
        if not all(isinstance(components.get(t), t) for t in required_types):
            print(f"GoalSystem: Entity {entity_id} missing essential components for goal update. Skipping.")
            return

        if not goal_comp.symbolic_goals_data:
            self._create_default_goals(goal_comp, current_tick, self.config)

        self._invent_and_refine_goals(entity_id, components, current_tick, self.config)
        best_goal = self._select_best_goal(entity_id, components, narrative, self.config)
        goal_comp.current_symbolic_goal = best_goal

    # ------------------------------------------------------------------ #
    # goal creation helpers
    # ------------------------------------------------------------------ #
    def _create_default_goals(self, goal_comp: GoalComponent, tick: int, cfg: Dict[str, Any]) -> None:
        emb_dim = get_config_value(cfg, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_cfg = cfg.get("llm", {})
        defaults = {"gather resources": "collecting materials", "survive": "prioritizing safety"}

        for name, descr in defaults.items():
            if name in goal_comp.symbolic_goals_data:
                continue
            emb = get_embedding_with_cache(f"{name}: {descr}", emb_dim, llm_cfg)
            if emb is not None:
                goal_comp.symbolic_goals_data[name] = {
                    "embedding": emb,
                    "success_history": [0.5],
                    "last_updated_tick": tick,
                }

    def _invent_and_refine_goals(
        self,
        eid: str,
        comps: Dict[Type[Component], Component],
        tick: int,
        cfg: Dict[str, Any],
    ) -> None:
        emb_dim = get_config_value(cfg, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_cfg = cfg.get("llm", {})

        mem_comp = cast(MemoryComponent, comps.get(MemoryComponent))
        goal_comp = cast(GoalComponent, comps.get(GoalComponent))
        if not mem_comp or not goal_comp:
            return

        recent = list(mem_comp.episodic_memory)[-30:]
        successful = [m for m in recent if m.get("outcome", 0) >= 0.1]
        if len(successful) < 5:
            return

        summaries = [
            f"{m['action'].get('action_type', 'NA')}->{m['outcome_details'].get('status', 'NA')}" for m in successful
        ]
        embeds = np.array([e for s in summaries if (e := get_embedding_with_cache(s, emb_dim, llm_cfg)) is not None])
        if len(embeds) < 3:
            return

        kmeans = KMeans(n_clusters=min(2, max(1, len(embeds) // 3)), random_state=tick, n_init="auto").fit(embeds)

        for i in range(kmeans.n_clusters):
            idxs = np.where(kmeans.labels_ == i)[0]
            if not idxs.size:
                continue
            sample = "; ".join(np.random.choice([summaries[j] for j in idxs], size=min(3, len(idxs)), replace=False))
            prompt = (
                f"These actions were successful: {sample}.\nWhat is a concise, 2-word goal that describes this pattern?"
            )
            try:
                goal_name = (
                    self.cognitive_scaffold.query(eid, "goal_invention", prompt, tick).strip().lower().replace(".", "")
                )
                if goal_name and len(goal_name) > 3:
                    self._add_goal_to_component(goal_comp, goal_name, len(idxs), tick, cfg)
            except Exception as exc:  # pragma: no cover
                print(f"Error inventing goal for {eid}: {exc}")

    def _add_goal_to_component(
        self, goal_comp: GoalComponent, name: str, success: int, tick: int, cfg: Dict[str, Any]
    ) -> None:
        emb_dim = get_config_value(cfg, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_cfg = cfg.get("llm", {})
        emb = get_embedding_with_cache(name, emb_dim, llm_cfg)
        if emb is None:
            return
        if name in goal_comp.symbolic_goals_data:
            goal_comp.symbolic_goals_data[name]["success_history"].extend([1.0] * success)
        else:
            goal_comp.symbolic_goals_data[name] = {
                "embedding": emb,
                "success_history": [1.0] * success,
                "last_updated_tick": tick,
            }

    # ------------------------------------------------------------------ #
    # goal selection
    # ------------------------------------------------------------------ #
    def _select_best_goal(
        self,
        eid: str,
        comps: Dict[Type[Component], Component],
        narrative: str,
        cfg: Dict[str, Any],
    ) -> Optional[str]:
        if self._is_in_critical_state(comps):
            print(f"GoalSystem: Entity {eid} in critical state. Forcing 'survive' goal.")
            return "survive"

        emb_dim = get_config_value(cfg, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        llm_cfg = cfg.get("llm", {})

        goal_comp = cast(GoalComponent, comps.get(GoalComponent))
        identity_comp = cast(IdentityComponent, comps.get(IdentityComponent))
        emotion_comp = cast(EmotionComponent, comps.get(EmotionComponent))
        if not all([goal_comp, identity_comp, emotion_comp]):
            return None

        context = f"My situation: {narrative}. My traits: {identity_comp.salient_traits_cache}. My emotion: {emotion_comp.current_emotion_category}."
        ctx_emb = get_embedding_with_cache(context, emb_dim, llm_cfg)
        if ctx_emb is None:
            return goal_comp.current_symbolic_goal

        id_emb = identity_comp.embedding
        best, best_score = None, -float("inf")
        for name, data in goal_comp.symbolic_goals_data.items():
            score = self._goal_score(data, ctx_emb, id_emb)
            if score > best_score:
                best, best_score = name, score

        print(f"GoalSystem: Entity {eid} chose goal '{best}' (score: {best_score:.3f})")
        return best

    def _goal_score(self, data: Dict[str, Any], ctx: np.ndarray, identity: Optional[np.ndarray]) -> float:
        emb = data.get("embedding")
        if emb is None:
            return -float("inf")
        ctx_sim = safe_cosine_similarity(ctx, emb)
        id_sim = safe_cosine_similarity(identity, emb) if identity is not None else 0.5
        success = float(np.mean(data.get("success_history", []))) if data.get("success_history") else 0.5
        return (ctx_sim * 2.0) + (success * 1.5) + (id_sim * 1.0)

    def _is_in_critical_state(self, comps: Dict[Type[Component], Component]) -> bool:
        health = cast(HealthComponent, comps.get(HealthComponent))
        time = cast(TimeBudgetComponent, comps.get(TimeBudgetComponent))
        inv = cast(InventoryComponent, comps.get(InventoryComponent))
        if not all([health, time, inv]):
            return False
        crit_cfg = self.config.get("learning", {}).get("critical_state", {})
        return (
            health.current_health <= health.initial_health * crit_cfg.get("health_threshold_percent", 0.2)
            or time.current_time_budget <= time.initial_time_budget * crit_cfg.get("time_threshold_percent", 0.1)
            or inv.current_resources <= crit_cfg.get("resource_threshold", 0.0)
        )

    # event-driven; nothing per-tick
    def update(self, current_tick: int) -> None:  # noqa: D401
        pass
