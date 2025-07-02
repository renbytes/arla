# src/systems/identity_system.py

from typing import Any, Dict, List, Optional, Type, cast

import numpy as np

# FIX: Import the concrete RelationalSchema and alias it to avoid name clashes.
from src.agents.components.relational_schema import RelationalSchema as RealRelationalSchema
from src.cognition.ai_models.openai_client import get_embedding_with_cache
from src.cognition.identity.domain_identity import IdentityDomain, SocialValidationCollector
from src.core.ecs.abstractions import CognitiveComponent
from src.core.ecs.component import IdentityComponent, SocialMemoryComponent
from src.core.ecs.event_bus import EventBus
from src.core.ecs.system import System
from src.utils.config_utils import get_config_value
from src.utils.math_utils import safe_normalize_vector


class IdentitySystem(System):
    """
    Updates an agent's multi-domain identity based on narrative reflections.
    This system is event-driven and implements the identity update model
    from Section 2.3 of the paper.
    """

    REQUIRED_COMPONENTS: List[Type[CognitiveComponent]] = []

    def __init__(self, simulation_state: Any, config: Dict[str, Any], cognitive_scaffold: Any):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.event_bus: EventBus = simulation_state.event_bus
        self.event_bus.subscribe("reflection_completed", self.on_reflection_completed)
        self.social_validator = SocialValidationCollector()

    def on_reflection_completed(self, event_data: Dict[str, Any]):
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

        llm_config = self.config.get("llm", {})
        inferred_traits_by_domain = self._infer_domain_traits_from_narrative(
            entity_id, narrative, current_tick, llm_config
        )
        if not inferred_traits_by_domain:
            return

        social_schemas_raw = social_memory_comp.schemas if isinstance(social_memory_comp, SocialMemoryComponent) else {}
        # FIX: Cast the dictionary to the specific type expected by the function.
        social_schemas = cast(Dict[str, RealRelationalSchema], social_schemas_raw)
        social_feedback = self.social_validator.collect_social_feedback(entity_id, {}, social_schemas, current_tick)

        self._apply_identity_updates(
            identity_comp,
            inferred_traits_by_domain,
            social_feedback,
            tick=current_tick,
            narrative=narrative,
        )

        self._log_identity_changes(entity_id, identity_comp)

    def _apply_identity_updates(
        self,
        identity_comp: IdentityComponent,
        inferred_traits_by_domain: Dict[IdentityDomain, Dict[str, float]],
        social_feedback: Dict[str, float],
        tick: int,
        narrative: str,
    ) -> None:
        """
        Calls the update method on the IdentityComponent for each domain.
        """
        llm_config = self.config.get("llm", {})
        embedding_dim = get_config_value(self.config, "agent.cognitive.embeddings.main_embedding_dim", default=1536)

        all_inferred_traits_cache = {}

        for domain, traits in inferred_traits_by_domain.items():
            if not traits:
                continue

            all_inferred_traits_cache.update(traits)

            trait_embeddings = []
            for name, strength in traits.items():
                embedding = get_embedding_with_cache(name, embedding_dim, llm_config)
                if embedding is not None:
                    trait_embeddings.append(embedding * strength)

            if not trait_embeddings:
                continue

            target_embedding_t = safe_normalize_vector(np.mean(trait_embeddings, axis=0))
            if target_embedding_t is None:
                continue

            identity_comp.update_domain_identity(
                domain=domain,
                new_traits=target_embedding_t,
                social_feedback=social_feedback,
                current_tick=tick,
                narrative_context=narrative,
            )

        identity_comp.salient_traits_cache = all_inferred_traits_cache

    def _infer_domain_traits_from_narrative(
        self, entity_id: str, narrative: str, current_tick: int, llm_config: Dict[str, Any]
    ) -> Dict[IdentityDomain, Dict[str, float]]:
        """
        Uses a structured prompt to have the LLM infer traits for each identity domain.
        """
        domain_definitions = """
                - SOCIAL: How I interact with others (e.g., friendly, hostile).
                - COMPETENCE: My skills and effectiveness (e.g., resourceful, clumsy).
                - MORAL: My ethical principles (e.g., fair, selfish).
                - RELATIONAL: My approach to close bonds (e.g., loyal, detached).
                - AGENCY: My sense of control and independence (e.g., decisive, follower).
        """
        llm_prompt = f"""Analyze the following narrative and identify 1-2 key traits for EACH of the five domains below. Assign each trait a score from 0.0 to 1.0 indicating its strength in the narrative.

                Narrative: "{narrative}"

                Domains:
                {domain_definitions}

                Format your response EXACTLY as follows, with each domain on a new line:
                SOCIAL:
                - trait_name: score
                COMPETENCE:
                - trait_name: score
                MORAL:
                - trait_name: score
                RELATIONAL:
                - trait_name: score
                AGENCY:
                - trait_name: score
                """
        try:
            raw_response = self.cognitive_scaffold.query(
                agent_id=entity_id, purpose="identity_domain_inference", prompt=llm_prompt, current_tick=current_tick
            )
            return self._parse_structured_llm_traits(raw_response)
        except Exception as e:
            print(f"Error in structured LLM trait inference for Entity {entity_id}: {e}")
            return {}

    def _parse_structured_llm_traits(self, llm_response: str) -> Dict[IdentityDomain, Dict[str, float]]:
        """
        Parses the structured response from the LLM.
        """
        # FIX: Added explicit type annotation
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

    def _log_identity_changes(self, entity_id: str, identity_comp: IdentityComponent):
        """Helper function to log significant identity-related events."""
        coherence = identity_comp.multi_domain_identity.get_identity_coherence()
        identity_config = self.config.get("agent", {}).get("identity_dynamics", {})
        coherence_minimum = identity_config.get("identity_coherence_minimum", 0.4)
        if coherence < coherence_minimum:
            print(f"Identity System: {entity_id} is experiencing an identity crisis (coherence: {coherence:.2f})")

    def update(self, current_tick: int):
        """This system is purely event-driven."""
        pass
