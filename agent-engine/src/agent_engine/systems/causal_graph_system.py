# FILE: agent-engine/src/agent_engine/systems/causal_graph_system.py

from typing import Any, Dict, List, Optional, Type, cast

import pandas as pd
from agent_core.agents.actions.action_interface import ActionInterface
from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import (
    ActionPlanComponent,
    Component,
    MemoryComponent,
    ValidationComponent,
)
from agent_core.core.ecs.event_bus import EventBus
from agent_core.environment.state_node_encoder_interface import (
    StateNodeEncoderInterface,
)
from dowhy import CausalModel

from agent_engine.cognition.reflection.validation import CausalModelValidator
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System


class CausalGraphSystem(System):
    """
    Constructs, validates, and maintains a formal causal model for each agent
    using the dowhy library.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [MemoryComponent, ValidationComponent]

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

        self._pre_action_states: Dict[str, tuple] = {}
        # Store the mapping from string category to integer code for each agent
        self._category_maps: Dict[str, Dict[str, Dict[str, int]]] = {}

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """Logs the (pre-state, action, outcome) tuple for model building."""
        entity_id = event_data["entity_id"]
        components = self.simulation_state.entities.get(entity_id, {})
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))

        if not mem_comp or entity_id not in self._pre_action_states:
            return

        pre_action_state_tuple = self._pre_action_states.pop(entity_id)
        action_plan = cast(ActionPlanComponent, event_data["action_plan"])
        action_outcome = cast(ActionOutcome, event_data["action_outcome"])

        record = self._flatten_data_for_record(
            pre_action_state_tuple, action_plan, action_outcome
        )
        if hasattr(mem_comp, "causal_data"):
            mem_comp.causal_data.append(record)

    async def update(self, current_tick: int) -> None:
        """Caches pre-action states and periodically rebuilds the causal model."""
        target_entities = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )
        for entity_id, components in target_entities.items():
            self._pre_action_states[entity_id] = (
                self.state_node_encoder.encode_state_for_causal_graph(
                    entity_id=entity_id,
                    components=components,
                    current_tick=current_tick,
                    config=self.config,
                )
            )

        # Rebuild model periodically
        rebuild_interval = self.config.learning.memory.get("reflection_interval", 50)
        if current_tick > 0 and current_tick % rebuild_interval == 0:
            for entity_id, components in target_entities.items():
                mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
                if (
                    mem_comp
                    and hasattr(mem_comp, "causal_data")
                    and len(mem_comp.causal_data) >= 20
                ):
                    self._build_and_validate_causal_model(entity_id, components)

    def estimate_causal_effect(
        self, agent_id: str, treatment_value: str
    ) -> Optional[float]:
        """Estimates the causal effect of a specific action on the outcome."""
        mem_comp = self.simulation_state.get_component(agent_id, MemoryComponent)
        agent_maps = self._category_maps.get(agent_id)

        if (
            not mem_comp
            or not hasattr(mem_comp, "causal_model")
            or not mem_comp.causal_model
            or not agent_maps
        ):
            return None

        # Convert the string action name into its integer code before querying
        action_map = agent_maps.get("action", {})
        treatment_code = action_map.get(treatment_value)
        if treatment_code is None:
            return None  # This action hasn't been seen/modeled yet

        try:
            identified_estimand = mem_comp.causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )
            estimate = mem_comp.causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate",
                method_params={"treatment_value": treatment_code},
            )
            return estimate.value
        except Exception:
            # This can happen if an action was observed but not enough to estimate
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

        flat_record = {"action": action_name, "outcome": outcome.reward}
        for i, value in enumerate(pre_state):
            if i > 0 and isinstance(value, str):
                parts = value.split("_", 1)
                if len(parts) == 2:
                    key, val = parts
                    flat_record[f"state_{key}"] = val
        return flat_record

    def _build_and_validate_causal_model(
        self, agent_id: str, components: Dict[Type[Component], Component]
    ) -> None:
        """Builds, validates, and stores a formal CausalModel for an agent."""
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        val_comp = cast(ValidationComponent, components.get(ValidationComponent))

        if not mem_comp or not val_comp or not hasattr(mem_comp, "causal_data"):
            return

        df = pd.DataFrame(mem_comp.causal_data)
        df.dropna(inplace=True)

        if len(df) < 20:
            return

        # Convert all categorical columns to integer codes
        self._category_maps[agent_id] = {}
        categorical_cols = ["action"] + [
            col for col in df.columns if col.startswith("state_")
        ]

        for col in categorical_cols:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category")
                self._category_maps[agent_id][col] = {
                    cat: i for i, cat in enumerate(df[col].cat.categories)
                }
                df[col] = df[col].cat.codes

        common_causes = [col for col in df.columns if col.startswith("state_")]
        if "action" not in df.columns or not common_causes:
            return

        try:
            model = CausalModel(
                data=df,
                treatment="action",
                outcome="outcome",
                common_causes=common_causes,
            )
            mem_comp.causal_model = model
            print(f"✅ Built new causal model for agent {agent_id}.")

            # Validate the model and store the confidence score
            validator = CausalModelValidator(model)
            results = validator.check_robustness()
            scores = [s for s in results.values() if s is not None]
            val_comp.causal_model_confidence = (
                sum(scores) / len(scores) if scores else 0.0
            )
            print(
                f"  - Causal model validation for {agent_id} complete. Confidence: {val_comp.causal_model_confidence:.2f}"
            )

        except Exception as e:
            print(f"❌ Error building causal model for {agent_id}: {e}")
            mem_comp.causal_model = None
            val_comp.causal_model_confidence = 0.0
