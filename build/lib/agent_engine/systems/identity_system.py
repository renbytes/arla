# src/agent_engine/systems/identity_system.py
"""
Updates an agent's multi-domain identity based on narrative reflections.
"""

from typing import Any, Dict, List, Optional, Type

import numpy as np

# Imports from agent_core
from agent_core.cognition.ai_models.openai_client import get_embedding_with_cache
from agent_core.cognition.identity.domain_identity import (
    IdentityDomain,
    SocialValidationCollector,
)
from agent_core.core.ecs.component import Component, IdentityComponent, SocialMemoryComponent
from agent_core.core.ecs.event_bus import EventBus

# Imports from agent_engine
from agent_engine.simulation.simulation_state import SimulationState
from agent_engine.simulation.system import System
from agent_engine.utils.config_utils import get_config_value
from agent_engine.utils.math_utils import safe_normalize_vector


class IdentitySystem(System):
    """
    Updates an agent's multi-domain identity based on narrative reflections.
    This system is event-driven and implements the identity update model.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Event-driven

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
    ) -> None:
        super().__init__(simulation_state, config, cognitive_scaffold)

        event_bus = simulation_state.event_bus
        if not event_bus:
            raise ValueError("EventBus not initialized in SimulationState")
        self.event_bus: EventBus = event_bus

        self.event_bus.subscribe("reflection_completed", self.on_reflection_completed)
        self.social_validator = SocialValidationCollector()

    def on_reflection_completed(self, event_data: Dict[str, Any]) -> None:
        """
        Event handler that orchestrates the full identity update cycle upon
        receiving a completed reflection from an agent.
        """
        entity_id = event_data["entity_id"]
        narrative = event_data["llm_final_account"]
        current_tick = event_data.get("tick", 0)

        components = self.simulation_state.entities.get(entity_id, {})
        identity_comp = components.get(IdentityComponent)
        social_memory_comp = components.get(SocialMemoryComponent)

        if not isinstance(identity_comp, IdentityComponent):
            return

        inferred_traits = self._infer_domain_traits_from_narrative(entity_id, narrative, current_tick)
        if not inferred_traits:
            return

        social_schemas = social_memory_comp.schemas if isinstance(social_memory_comp, SocialMemoryComponent) else {}
        social_feedback = self.social_validator.collect_social_feedback(entity_id, {}, social_schemas, current_tick)

        self._apply_identity_updates(
            identity_comp,
            inferred_traits,
            social_feedback,
            tick=current_tick,
            narrative=narrative,
        )

    def _apply_identity_updates(
        self,
        identity_comp: IdentityComponent,
        inferred_traits_by_domain: Dict[IdentityDomain, Dict[str, float]],
        social_feedback: Dict[str, float],
        tick: int,
        narrative: str,
    ) -> None:
        """Calls the update method on the IdentityComponent for each domain."""
        llm_config = self.config.get("llm", {})
        embedding_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", 1536)
        all_inferred_traits_cache = {}

        for domain, traits in inferred_traits_by_domain.items():
            if not traits:
                continue

            all_inferred_traits_cache.update(traits)
            trait_embeddings = [
                embedding * strength
                for name, strength in traits.items()
                if (embedding := get_embedding_with_cache(name, embedding_dim, llm_config)) is not None
            ]

            if not trait_embeddings:
                continue

            target_embedding = safe_normalize_vector(np.mean(trait_embeddings, axis=0))
            if target_embedding is None:
                continue

            identity_comp.multi_domain_identity.update_domain_identity(
                domain=domain,
                new_traits=target_embedding,
                social_feedback=social_feedback,
                current_tick=tick,
                narrative_context=narrative,
            )
        identity_comp.salient_traits_cache = all_inferred_traits_cache

    def _infer_domain_traits_from_narrative(
        self, entity_id: str, narrative: str, current_tick: int
    ) -> Dict[IdentityDomain, Dict[str, float]]:
        """Uses a structured prompt to have the LLM infer traits for each identity domain."""
        domain_definitions = "\n".join([f"- {d.value.upper()}: ..." for d in IdentityDomain])
        llm_prompt = f"""Analyze the narrative below and identify 1-2 key traits for EACH domain. Assign each trait a score from 0.0 to 1.0.

Narrative: "{narrative}"

Domains:
{domain_definitions}

Format your response EXACTLY as follows:
SOCIAL:
- trait_name: score
COMPETENCE:
- trait_name: score
...
"""
        try:
            raw_response = self.cognitive_scaffold.query(
                agent_id=entity_id,
                purpose="identity_domain_inference",
                prompt=llm_prompt,
                current_tick=current_tick,
            )
            return self._parse_structured_llm_traits(raw_response)
        except Exception as e:
            print(f"Error in structured LLM trait inference for Entity {entity_id}: {e}")
            return {}

    def _parse_structured_llm_traits(self, llm_response: str) -> Dict[IdentityDomain, Dict[str, float]]:
        """Parses the structured response from the LLM."""
        domain_traits: Dict[IdentityDomain, Dict[str, float]] = {domain: {} for domain in IdentityDomain}
        current_domain: Optional[IdentityDomain] = None
        for line in llm_response.splitlines():
            line = line.strip()
            if not line:
                continue

            potential_domain_str = line.split(":")[0].upper()
            try:
                current_domain = IdentityDomain[potential_domain_str]
                continue
            except KeyError:
                pass

            if current_domain and line.startswith("-"):
                try:
                    name, score_str = line[1:].split(":")
                    trait_name = name.strip().lower().replace(" ", "_")
                    score = np.clip(float(score_str.strip()), 0.0, 1.0)
                    domain_traits[current_domain][trait_name] = score
                except (ValueError, IndexError):
                    continue
        return domain_traits

    def update(self, current_tick: int) -> None:
        """This system is purely event-driven."""
        pass
