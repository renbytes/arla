# src/agent_core/cognition/identity/domain_identity.py

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, cast

import numpy as np
from agent_core.core.ecs.component import MultiDomainIdentityInterface

if TYPE_CHECKING:
    # This will be defined in the agent-engine or the final simulation application
    from agent_core.core.schemas import RelationalSchema


class IdentityDomain(Enum):
    """Core identity domains from social psychology literature"""

    SOCIAL = "social"  # How I relate to others
    COMPETENCE = "competence"  # What I'm good at
    MORAL = "moral"  # My values and ethics
    RELATIONAL = "relational"  # Close relationships and attachments
    AGENCY = "agency"  # Personal control and autonomy


@dataclass
class DomainIdentity:
    """Identity representation for a specific domain"""

    domain: IdentityDomain
    embedding: np.ndarray
    confidence: float  # How certain am I about this aspect?
    stability: float  # How resistant to change?
    social_validation: float  # How much social confirmation?
    last_updated: int  # When was this last changed?


class MultiDomainIdentity(MultiDomainIdentityInterface):
    """Psychologically-grounded multi-domain identity system

    Note:
        embedding_dim is defaulted to `1536` as that's the standard dim set by OpenAI.
    """

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.domains: Dict[IdentityDomain, DomainIdentity] = {}
        self.global_consistency_threshold = 0.7
        self.social_validation_weight = 0.6

        # Initialize domains with neutral embeddings
        for domain in IdentityDomain:
            self.domains[domain] = DomainIdentity(
                domain=domain,
                embedding=np.random.normal(0, 0.1, embedding_dim).astype(np.float32),
                confidence=0.3,  # Start uncertain
                stability=0.25,  # Moderate resistance to change
                social_validation=0.0,
                last_updated=0,
            )

    def update_domain_identity(
        self,
        domain: IdentityDomain,
        new_traits: np.ndarray,
        context: Dict[str, Any],
        current_tick: int,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Update identity in a specific domain with psychological constraints
        Returns: (update_occurred, consistency_score, validation_metrics)
        """
        current_domain = self.domains[domain]

        consistency_score = self._assess_consistency(new_traits, current_domain.embedding, current_domain.confidence)
        validation_score = self._assess_social_validation(context, domain)
        resistance = self._calculate_resistance(current_domain, consistency_score, validation_score)

        update_threshold = 0.3 + (current_domain.stability * 0.4)
        combined_support = consistency_score * 0.4 + validation_score * 0.6

        update_occurred = False
        if combined_support > update_threshold:
            update_strength = self._calculate_update_strength(
                consistency_score, validation_score, resistance, current_domain.confidence
            )
            identity_delta = new_traits - current_domain.embedding
            constrained_delta = identity_delta * update_strength * (1 - resistance)

            current_domain.embedding = current_domain.embedding + constrained_delta
            current_domain.embedding = self._normalize_embedding(current_domain.embedding)

            current_domain.confidence = min(1.0, current_domain.confidence + 0.1)
            current_domain.social_validation = validation_score
            current_domain.last_updated = current_tick

            if combined_support > 0.8:
                current_domain.stability = min(1.0, current_domain.stability + 0.05)
            update_occurred = True

        validation_metrics = {
            "consistency": consistency_score,
            "social_validation": validation_score,
            "resistance": resistance,
            "combined_support": combined_support,
            "update_threshold": update_threshold,
        }

        return update_occurred, consistency_score, validation_metrics

    def _assess_consistency(
        self, new_traits: np.ndarray, current_embedding: np.ndarray, current_confidence: float
    ) -> float:
        """Assess how consistent new traits are with existing self-concept."""
        if current_confidence < 0.1:
            return 0.8

        norm_new = np.linalg.norm(new_traits)
        norm_current = np.linalg.norm(current_embedding)

        if norm_new == 0 or norm_current == 0:
            return 0.5

        similarity = np.dot(new_traits, current_embedding) / (norm_new * norm_current)
        consistency = (similarity + 1.0) / 2.0
        confidence_modulated = consistency ** (1 + current_confidence)
        return float(np.clip(confidence_modulated, 0.0, 1.0))

    def _assess_social_validation(self, context: Dict[str, Any], domain: IdentityDomain) -> float:
        """Assess social validation for identity claims."""
        social_feedback = context.get("social_feedback", {})
        if not social_feedback:
            return 0.3

        domain_weights = {
            IdentityDomain.SOCIAL: 0.9,
            IdentityDomain.COMPETENCE: 0.6,
            IdentityDomain.MORAL: 0.4,
            IdentityDomain.RELATIONAL: 0.8,
            IdentityDomain.AGENCY: 0.3,
        }
        base_validation = domain_weights.get(domain, 0.5)

        positive_interactions = social_feedback.get("positive_social_responses", 0.0)
        negative_interactions = social_feedback.get("negative_social_responses", 0.0)
        social_approval = social_feedback.get("social_approval_rating", 0.0)
        peer_recognition = social_feedback.get("peer_recognition", 0.0)

        validation_score = (
            positive_interactions * 0.3 + social_approval * 0.4 + peer_recognition * 0.3 - negative_interactions * 0.2
        )
        final_validation = base_validation * 0.3 + validation_score * 0.7
        return float(np.clip(final_validation, 0.0, 1.0))

    def _calculate_resistance(
        self, current_domain: DomainIdentity, consistency_score: float, validation_score: float
    ) -> float:
        """Calculate psychological resistance to identity change."""
        stability_resistance = current_domain.stability * 0.4
        consistency_resistance = (1 - consistency_score) * 0.5
        validation_resistance = (1 - validation_score) * 0.3
        confidence_resistance = current_domain.confidence * 0.2
        total_resistance = stability_resistance + consistency_resistance + validation_resistance + confidence_resistance
        return float(np.clip(total_resistance, 0.0, 0.9))

    def _calculate_update_strength(
        self, consistency_score: float, validation_score: float, resistance: float, current_confidence: float
    ) -> float:
        """Calculate how strongly to update the identity embedding."""
        evidence_strength = consistency_score * 0.4 + validation_score * 0.6
        resistance_modulated = evidence_strength * (1 - resistance)
        confidence_modulated = resistance_modulated * (1.2 - current_confidence)
        return float(np.clip(confidence_modulated, 0.0, 0.3))

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding

        return cast(np.ndarray, embedding / norm)

    def get_domain_embedding(self, domain: IdentityDomain) -> np.ndarray:
        """Get embedding for a specific identity domain."""
        return self.domains[domain].embedding.copy()

    def get_global_identity_embedding(self) -> np.ndarray:
        """Compute global identity as a confidence-weighted average of domains."""
        total_weight = 0.0
        weighted_sum = np.zeros(self.embedding_dim)

        for domain_identity in self.domains.values():
            weight = domain_identity.confidence
            weighted_sum += domain_identity.embedding * weight
            total_weight += weight

        if total_weight == 0:
            return np.zeros(self.embedding_dim)

        return cast(np.ndarray, weighted_sum / total_weight)

    def get_identity_coherence(self) -> float:
        """Measure how coherent the identity is across domains."""
        embeddings = [d.embedding for d in self.domains.values()]
        confidences = [d.confidence for d in self.domains.values()]

        if len(embeddings) < 2:
            return 1.0

        similarities = []
        weights = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                emb1, emb2 = embeddings[i], embeddings[j]
                conf1, conf2 = confidences[i], confidences[j]
                norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                    similarities.append((similarity + 1) / 2)
                    weights.append(conf1 * conf2)

        if not similarities:
            return 1.0

        total_weight: float = sum(weights)
        if total_weight == 0:
            return float(np.mean(similarities))

        weighted_coherence = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        return float(weighted_coherence)

    def get_identity_stability(self) -> float:
        """Get overall identity stability across domains"""
        stabilities = [d.stability for d in self.domains.values()]
        return float(np.mean(stabilities)) if stabilities else 0.5


class SocialValidationCollector:
    """Collects and processes social feedback for identity validation."""

    def __init__(self) -> None:
        self.validation_history: Dict[str, List[Dict[str, float]]] = {}

    def collect_social_feedback(
        self,
        entity_id: str,
        interaction_data: Dict[str, Any],
        social_schemas: Dict[str, "RelationalSchema"],
        current_tick: int,
    ) -> Dict[str, float]:
        """Collect social validation signals from recent interactions."""
        feedback = {
            "positive_social_responses": 0.0,
            "negative_social_responses": 0.0,
            "social_approval_rating": 0.0,
            "peer_recognition": 0.0,
            "interaction_frequency": 0.0,
        }
        if not social_schemas:
            return feedback

        positive_count, negative_count, total_valence, recognition_score = 0, 0, 0.0, 0.0

        for schema in social_schemas.values():
            if hasattr(schema, "impression_valence") and hasattr(schema, "interaction_count"):
                valence = schema.impression_valence
                interactions = schema.interaction_count
                total_valence += valence
                if valence > 0.3:
                    positive_count += interactions
                elif valence < -0.3:
                    negative_count += interactions
                recognition_score += min(interactions / 10.0, 1.0)

        total_interactions = positive_count + negative_count
        if total_interactions > 0:
            feedback["positive_social_responses"] = positive_count / total_interactions
            feedback["negative_social_responses"] = negative_count / total_interactions

        if social_schemas:
            feedback["social_approval_rating"] = np.clip((total_valence / len(social_schemas) + 1) / 2, 0, 1)
            feedback["peer_recognition"] = min(recognition_score / len(social_schemas), 1.0)
            feedback["interaction_frequency"] = min(len(social_schemas) / 5.0, 1.0)

        if entity_id not in self.validation_history:
            self.validation_history[entity_id] = []
        self.validation_history[entity_id].append({**feedback, "tick": current_tick})
        if len(self.validation_history[entity_id]) > 20:
            self.validation_history[entity_id] = self.validation_history[entity_id][-20:]

        return feedback
