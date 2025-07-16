# simulations/emergence_sim/systems/ritualization_system.py
"""
This system identifies and codifies ritualized behaviors by correlating
action sequences with reductions in cognitive dissonance, based on the
neuro-cognitive model of a "Hazard-Precaution System".
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from agent_core.core.ecs.component import (
    ActionPlanComponent,
    AffectComponent,
    Component,
    MemoryComponent,
)
from agent_engine.simulation.system import System
from components import RitualComponent


class RitualizationSystem(System):
    """
    Identifies and codifies ritualized behaviors by monitoring agent state.
    This system periodically scans an agent's memory for patterns where a
    negative event (a "trigger") is consistently followed by a specific,
    stereotyped sequence of actions that lead to positive outcomes.
    This
    pattern suggests the sequence is being used to recover from the negative
    state, akin to a ritual.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = [
        MemoryComponent,
        RitualComponent,
        AffectComponent,
        ActionPlanComponent,
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Use direct attribute access on the validated Pydantic config model
        self.dissonance_threshold: float = self.config.learning.memory.cognitive_dissonance_threshold
        self.ritual_codification_threshold: int = 3
        self.ritual_sequence_length: int = 2

    async def update(self, current_tick: int) -> None:
        """
        Periodically scans agent memories to find and codify new rituals.
        This runs infrequently to reduce computational load.
        """
        # Use direct attribute access on the validated Pydantic config model
        reflection_interval = self.config.learning.memory.reflection_interval
        if current_tick == 0 or current_tick % reflection_interval != 0:
            return

        all_relevant_entities = self.simulation_state.get_entities_with_components(self.REQUIRED_COMPONENTS)

        for entity_id, components in all_relevant_entities.items():
            await self._analyze_and_codify_rituals_for_agent(entity_id, components, current_tick)

    async def _analyze_and_codify_rituals_for_agent(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        current_tick: int,
    ) -> None:
        """Analyzes a single agent's memory for potential rituals by orchestrating helper methods."""
        mem_comp = cast(MemoryComponent, components.get(MemoryComponent))
        if not mem_comp or len(mem_comp.episodic_memory) < self.ritual_sequence_length + 1:
            return

        # Step 1: Find all sequences that follow a negative event
        candidate_sequences = self._find_candidate_sequences(mem_comp)
        if not candidate_sequences:
            return

        # Step 2: Identify and codify the most common, successful sequences
        self._codify_widespread_rituals(entity_id, components, candidate_sequences)

    def _find_candidate_sequences(self, mem_comp: MemoryComponent) -> Dict[str, List[Tuple[str, ...]]]:
        """Iterates through memory to find all valid, successful action sequences following a trigger."""
        candidates: Dict[str, List[Tuple[str, ...]]] = {}
        for i in range(len(mem_comp.episodic_memory) - self.ritual_sequence_length):
            trigger_event = mem_comp.episodic_memory[i]
            trigger_type = self._identify_trigger(trigger_event)

            if trigger_type:
                sequence = self._extract_valid_sequence_after_trigger(mem_comp, i)
                if sequence:
                    if trigger_type not in candidates:
                        candidates[trigger_type] = []
                    candidates[trigger_type].append(sequence)
        return candidates

    def _extract_valid_sequence_after_trigger(
        self, mem_comp: MemoryComponent, start_index: int
    ) -> Optional[Tuple[str, ...]]:
        """Extracts and validates a single action sequence from memory."""
        sequence: List[str] = []
        for i in range(1, self.ritual_sequence_length + 1):
            event = mem_comp.episodic_memory[start_index + i]
            action_plan = event.get("action_plan")

            # A sequence is only valid if every action in it was successful (positive reward)
            if event.get("reward", 0.0) < 0 or not action_plan or not hasattr(action_plan, "action_type"):
                return None  # Invalid sequence

            action_id = getattr(action_plan.action_type, "action_id", "unknown")
            sequence.append(action_id)

        return tuple(sequence)

    def _codify_widespread_rituals(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        candidates: Dict[str, List[Tuple[str, ...]]],
    ) -> None:
        """Counts candidate sequences and codifies the most common ones as rituals."""
        ritual_comp = cast(RitualComponent, components[RitualComponent])

        for trigger, sequences in candidates.items():
            if not sequences:
                continue

            sequence_counts = Counter(sequences)
            most_common_sequence, count = sequence_counts.most_common(1)[0]

            if count >= self.ritual_codification_threshold and trigger not in ritual_comp.codified_rituals:
                ritual_comp.codified_rituals[trigger] = list(most_common_sequence)
                print(f"INFO: Agent {entity_id} codified new ritual '{trigger}': {most_common_sequence}")

    def _identify_trigger(self, event: Dict[str, Any]) -> Optional[str]:
        """
        Identifies if an event constitutes a trigger for ritualization.
        Here, a trigger is a highly negative event.
        """
        reward = event.get("reward", 0.0)
        action_plan = event.get("action_plan")
        action_id = "unknown"
        if action_plan and hasattr(action_plan, "action_type"):
            action_id = getattr(action_plan.action_type, "action_id", "unknown")

        # Example trigger: losing a fight
        if action_id == "combat" and reward < -5.0:
            return "post_combat_loss"

        # Example trigger: failing to extract resources
        if action_id == "extract" and reward < -1.0:
            return "post_failed_extraction"

        return None
