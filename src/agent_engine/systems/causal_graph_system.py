# src/systems/causal_graph_system.py

from typing import Any, Dict, List, Tuple, Type, cast

from src.agents.actions.base_action import ActionOutcome
from src.cognition.memory.symbolic_graph import (
    _create_action_outcome_node,
    _create_state_node,
    add_causal_link,
    decay_causal_links,
)
from src.core.ecs.abstractions import CognitiveComponent
from src.core.ecs.component import (
    ActionPlanComponent,
    EmotionComponent,
    GoalComponent,
    HealthComponent,
    InventoryComponent,
    MemoryComponent,
    PositionComponent,
    TimeBudgetComponent,
)
from src.core.ecs.event_bus import EventBus
from src.core.ecs.system import System


class CausalGraphSystem(System):
    """
    Constructs and maintains a symbolic causal graph for each agent's memory.
    Refactored for dependency injection.
    """

    # The per-tick update loop requires entities with a MemoryComponent to decay the graph.
    REQUIRED_COMPONENTS: List[Type[CognitiveComponent]] = [MemoryComponent]

    def __init__(self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus: EventBus = simulation_state.event_bus
        self.event_bus.subscribe("action_executed", self.on_action_executed)
        # Caches the previous state node for each entity to link actions correctly
        self.previous_state_nodes: Dict[str, Tuple[str, ...]] = {}

    def on_action_executed(self, event_data: Dict[str, Any]):
        """Event handler that fetches dependencies and injects them into the graph logic."""
        entity_id = event_data["entity_id"]
        components = self.simulation_state.entities.get(entity_id, {})

        # --- Use isinstance for explicit type narrowing ---
        mem_comp = components.get(MemoryComponent)
        if not isinstance(mem_comp, MemoryComponent):
            return
        pos_comp = components.get(PositionComponent)
        if not isinstance(pos_comp, PositionComponent):
            return
        health_comp = components.get(HealthComponent)
        if not isinstance(health_comp, HealthComponent):
            return
        time_comp = components.get(TimeBudgetComponent)
        if not isinstance(time_comp, TimeBudgetComponent):
            return
        inv_comp = components.get(InventoryComponent)
        if not isinstance(inv_comp, InventoryComponent):
            return
        emotion_comp = components.get(EmotionComponent)
        if not isinstance(emotion_comp, EmotionComponent):
            return
        goal_comp = components.get(GoalComponent)
        if not isinstance(goal_comp, GoalComponent):
            return

        # --- Pass specific, validated components to the logic function ---
        self._update_causal_link(
            entity_id=entity_id,
            mem_comp=mem_comp,
            pos_comp=pos_comp,
            health_comp=health_comp,
            time_comp=time_comp,
            inv_comp=inv_comp,
            emotion_comp=emotion_comp,
            goal_comp=goal_comp,
            action_plan=event_data["action_plan"],
            action_outcome=event_data["action_outcome"],
        )

    def _update_causal_link(
        self,
        entity_id: str,
        mem_comp: MemoryComponent,
        pos_comp: PositionComponent,
        health_comp: HealthComponent,
        time_comp: TimeBudgetComponent,
        inv_comp: InventoryComponent,
        emotion_comp: EmotionComponent,
        goal_comp: GoalComponent,
        action_plan: ActionPlanComponent,
        action_outcome: ActionOutcome,
    ):
        """Pure logic for creating and linking nodes in an agent's causal graph."""
        try:
            # All components are now guaranteed to be the correct subclass.
            critical_state_config = self.config.get("learning", {}).get("critical_state", {})
            health_threshold = critical_state_config.get("health_threshold_percent", 0.2)
            is_in_danger = health_comp.current_health <= health_comp.initial_health * health_threshold

            state_snapshot = {
                "health": health_comp.current_health,
                "original_health": health_comp.initial_health,
                "time_budget": time_comp.current_time_budget,
                "resource_inventory": inv_comp.current_resources,
                "current_emotion_valence": emotion_comp.valence,
            }

            # FIX: Provide a default value for the symbolic goal if it is None.
            symbolic_goal = goal_comp.current_symbolic_goal or "no_goal"
            current_state_node = _create_state_node(state_snapshot, pos_comp.position, is_in_danger, symbolic_goal)

            # FIX: Safely access action_type.name, handling the case where action_type is None.
            action_type_name = "unknown_action"
            if action_plan.action_type:
                action_type_name = getattr(action_plan.action_type, "name", str(action_plan.action_type))

            intent_name = action_plan.intent.name if action_plan.intent else "UNKNOWN_INTENT"
            action_outcome_node = _create_action_outcome_node(
                action_type_name, intent_name, action_outcome.details, action_outcome.reward
            )

            link_weight = (abs(action_outcome.reward) * 0.1) + (emotion_comp.arousal * 0.5) + 0.1
            if previous_node := self.previous_state_nodes.get(entity_id):
                add_causal_link(mem_comp.causal_graph, previous_node, action_outcome_node, weight=link_weight)

            add_causal_link(mem_comp.causal_graph, action_outcome_node, current_state_node, weight=link_weight)

            self.previous_state_nodes[entity_id] = current_state_node
            pos_comp.history.append(pos_comp.position)

        except Exception as e:
            print(f"CausalGraphSystem Error for {entity_id}: {e}")

    def update(self, current_tick: int):
        """
        Periodically decays the strength of all causal links in every agent's memory.
        """
        if (current_tick + 1) % 10 == 0:
            # Use type: ignore to handle the list variance, which is a known mypy constraint.
            target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)  # type: ignore[arg-type]

            for components in target_entities.values():
                # Use cast to inform mypy of the specific component type.
                mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
                if mem_comp:
                    decay_causal_links(mem_comp.causal_graph, decay_rate=0.95)
