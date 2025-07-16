# agent-engine/src/agent_engine/systems/normative_abstraction_system.py
"""
This system provides a mechanism for computational metacognition, allowing the
simulation to "observe" its own emergent social dynamics and create abstract,
symbolic labels for them.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Type, cast

from agent_core.core.ecs.component import Component
from agent_engine.simulation.system import System

# This system is part of the generic engine but is designed to operate on
# specific components from the 'emergence_sim'. This is a form of dependency
# injection where the system is generic in principle, but its application here
# is specific to the research roadmap. We use a soft import to handle cases
# where the simulation might be run without these specific components.
try:
    from simulations.emergence_sim.components import (
        RitualComponent,
        SocialCreditComponent,
    )

    EMERGENCE_COMPONENTS_LOADED = True
except ImportError:
    EMERGENCE_COMPONENTS_LOADED = False


class NormativeAbstractionSystem(System):
    """
    A high-level, singleton system that observes the collective social state,
    identifies emergent patterns (norms), and uses the CognitiveScaffold to
    generate abstract labels for them. This enables agents to reason about
    the social structures they have created.
    """

    # This system operates on the global state, not on a per-entity basis,
    # so it does not require specific components on each entity it processes.
    REQUIRED_COMPONENTS: List[Type[Component]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the system with configurable thresholds."""
        super().__init__(*args, **kwargs)
        # Use direct attribute access on the validated Pydantic model
        self.abstraction_interval: int = 100  # How often to check for norms
        self.reciprocity_threshold: float = 0.7  # Avg social credit to define reciprocity
        self.ritual_adoption_threshold: float = 0.5  # Pct of agents needed to form a ritual norm
        # Keep track of norms that have already been named to avoid repetition
        self.defined_norms: set[str] = set()

    async def update(self, current_tick: int) -> None:
        """
        Periodically scans the entire simulation state for emergent social patterns
        and, if found, initiates the process of abstracting them into named norms.
        """
        if current_tick == 0 or current_tick % self.abstraction_interval != 0 or not EMERGENCE_COMPONENTS_LOADED:
            return

        print(f"--- [{current_tick}] Normative Abstraction System Running ---")

        # --- Check for Reciprocity Norm ---
        reciprocity_summary = self._detect_reciprocity_norm()
        if reciprocity_summary and "reciprocity" not in self.defined_norms:
            await self._abstract_and_publish_norm("reciprocity", reciprocity_summary, current_tick)

        # --- Check for Ritual Norms ---
        ritual_summaries = self._detect_ritual_norms()
        for ritual_name, summary in ritual_summaries.items():
            if ritual_name not in self.defined_norms:
                await self._abstract_and_publish_norm(ritual_name, summary, current_tick)

    def _detect_reciprocity_norm(self) -> Optional[str]:
        """
        Detects a norm of reciprocity by checking the average social credit score
        across the entire population. A high average score implies that pro-social,
        reciprocal behaviors are widespread and rewarded.
        """
        all_agents_with_credit = self.simulation_state.get_entities_with_components(SocialCreditComponent)
        if not all_agents_with_credit:
            return None

        scores = [
            cast(SocialCreditComponent, comps[SocialCreditComponent]).score for comps in all_agents_with_credit.values()
        ]

        if not scores:
            return None

        avg_score = sum(scores) / len(scores)
        if avg_score > self.reciprocity_threshold:
            return (
                f"Agents consistently engage in mutually beneficial exchanges, "
                f"leading to a high average social credit score of {avg_score:.2f}."
            )
        return None

    def _detect_ritual_norms(self) -> Dict[str, str]:
        """
        Detects dominant ritualistic norms by finding which codified rituals
        are shared by a significant portion of the agent population.
        """
        all_agents_with_rituals = self.simulation_state.get_entities_with_components(RitualComponent)
        if not all_agents_with_rituals:
            return {}

        all_rituals: List[str] = []
        for components in all_agents_with_rituals.values():
            ritual_comp = cast(RitualComponent, components[RitualComponent])
            all_rituals.extend(ritual_comp.codified_rituals.keys())

        if not all_rituals:
            return {}

        widespread_rituals = {}
        ritual_counts = Counter(all_rituals)
        num_agents = len(all_agents_with_rituals)

        for ritual_name, count in ritual_counts.items():
            adoption_rate = count / num_agents
            if adoption_rate >= self.ritual_adoption_threshold:
                summary = (
                    f"A significant portion of the population ({adoption_rate:.0%}) "
                    f"has adopted a specific ritual sequence in response to '{ritual_name}' events."
                )
                widespread_rituals[ritual_name] = summary

        return widespread_rituals

    async def _abstract_and_publish_norm(self, norm_key: str, summary: str, tick: int) -> None:
        """
        Uses the CognitiveScaffold (LLM) to generate a concise name for an
        observed social pattern and publishes it to the event bus for all
        agents to "learn".
        """
        print(f"INFO: Detected potential new norm: {norm_key}")
        prompt = f"""
        The following social behavior is consistently observed in a population of agents:
        '{summary}'

        Provide a single, abstract, one-word noun that best describes this social rule or value.
        For example: Reciprocity, Purity, Fairness, Vengeance.
        Respond with only the single word.
        """

        try:
            # The 'agent_id' is 'system' as this is a global, meta-level observation.
            abstract_norm_label = (
                self.cognitive_scaffold.query(
                    agent_id="system",
                    purpose="normative_abstraction",
                    prompt=prompt,
                    current_tick=tick,
                )
                .strip()
                .capitalize()
            )

            if abstract_norm_label and " " not in abstract_norm_label:
                print(f"INFO: Abstracted norm '{norm_key}' as '{abstract_norm_label}'.")
                # Publish the new abstract concept for other systems to process
                if self.event_bus:
                    self.event_bus.publish(
                        "abstract_norm_defined",
                        {"label": abstract_norm_label, "source_key": norm_key},
                    )
                # Add to the set of defined norms to prevent re-labeling
                self.defined_norms.add(norm_key)
            else:
                print(f"WARNING: LLM produced invalid norm label: '{abstract_norm_label}'")

        except Exception as e:
            print(f"ERROR: LLM query failed for normative abstraction: {e}")
