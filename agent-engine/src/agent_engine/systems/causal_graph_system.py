# src/agent_engine/systems/causal_graph_system.py
"""
Constructs and maintains a formal causal model for each agent's memory using the dowhy library,
and provides methods for causal inference using do-calculus.
"""

from typing import Any, Dict, List, Optional, Type, cast

import pandas as pd
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    EmotionComponent,
    GoalComponent,
    MemoryComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from agent_core.environment.state_node_encoder_interface import StateNodeEncoderInterface
from dowhy import CausalModel

from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


class CausalGraphSystem(System):
    """
    Constructs and maintains a formal causal model for each agent using `dowhy`.
    This system collects data from agent experiences, periodically builds a
    CausalModel, and exposes methods for estimating causal effects.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [MemoryComponent, EmotionComponent, GoalComponent]

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Any,
        cognitive_scaffold: Any,
        state_node_encoder: StateNodeEncoderInterface,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.state_node_encoder = state_node_encoder
        self.event_bus: Optional[EventBus] = self.simulation_state.event_bus
        if self.event_bus:
            self.event_bus.subscribe("action_executed", self.on_action_executed)

        self.causal_graph_dot = """
        digraph {
            U [label="Unobserved Confounders"];
            state -> action;
            action -> outcome;
            state -> outcome;
            U -> state;
            U -> outcome;
        }
        """
        self._pre_action_states: Dict[str, tuple] = {}

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """
        Event handler that logs the (pre-state, action, outcome, event_id)
        tuple required for causal model building.
        """
        entity_id = event_data["entity_id"]
        components = self.simulation_state.entities.get(entity_id, {})
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent, None))

        if not mem_comp or entity_id not in self._pre_action_states:
            return

        pre_action_state_tuple = self._pre_action_states.pop(entity_id)
        action_plan = cast(ActionPlanComponent, event_data["action_plan"])
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])

        record = self._flatten_data_for_record(pre_action_state_tuple, action_plan, action_outcome)
        if hasattr(mem_comp, "causal_data"):
            mem_comp.causal_data.append(record)

    async def update(self, current_tick: int) -> None:
        """
        Periodically rebuilds the causal model for each agent using the data
        collected in their MemoryComponent.
        """
        target_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)
        for entity_id, components in target_entities.items():
            self._pre_action_states[entity_id] = self.state_node_encoder.encode_state_for_causal_graph(
                entity_id=entity_id,
                components=components,
                current_tick=current_tick,
                config=self.config,
            )

        if current_tick > 0 and current_tick % 50 == 0:
            for _entity_id, components in target_entities.items():
                mem_comp = cast(MemoryComponent, components.get(MemoryComponent, None))
                if mem_comp and hasattr(mem_comp, "causal_data") and len(mem_comp.causal_data) >= 20:
                    self._build_causal_model(mem_comp)

    def estimate_causal_effect(self, agent_id: str, treatment_value: str) -> Optional[float]:
        """
        Estimates the causal effect of a specific action (treatment) on the outcome
        for a given agent using do-calculus.
        """
        mem_comp = self.simulation_state.get_component(agent_id, MemoryComponent)
        if not mem_comp or not hasattr(mem_comp, "causal_model") or not mem_comp.causal_model:
            return None

        causal_model = mem_comp.causal_model
        try:
            identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
            estimate = causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate",
                method_params={"treatment_value": treatment_value},
            )
            return estimate.value
        except Exception as e:
            print(f"Failed to estimate causal effect for {agent_id} due to: {e}")
            return None

    def _flatten_data_for_record(
        self,
        pre_state: tuple,
        action_plan: ActionPlanComponent,
        outcome: ActionOutcome,
    ) -> Dict[str, Any]:
        """Converts nested data into a flat dictionary for a DataFrame row."""
        action_name = "unknown"
        if isinstance(action_plan.action_type, ActionInterface):
            action_name = action_plan.action_type.action_id

        # NEW: Extract the event_id from the outcome details
        event_id = outcome.details.get("event_id", None)

        return {
            "event_id": event_id,  # Store the unique ID
            "state_health": pre_state[1],
            "state_location": pre_state[2],
            "action": action_name,
            "outcome": outcome.reward,
        }

    def _build_causal_model(self, mem_comp: MemoryComponent) -> None:
        """Uses the collected data to build and store a formal CausalModel."""
        if not hasattr(mem_comp, "causal_data") or not mem_comp.causal_data:
            return

        df = pd.DataFrame(mem_comp.causal_data)
        if df.empty:
            return

        common_causes = [col for col in df.columns if col.startswith("state_")]

        try:
            model = CausalModel(
                data=df,
                treatment="action",
                outcome="outcome",
                graph=self.causal_graph_dot,
                common_causes=common_causes,
            )
            mem_comp.causal_model = model
        except Exception as e:
            print(f"Error building causal model: {e}")
            mem_comp.causal_model = None
