from typing import Any, Dict, Optional

import numpy as np
from agent_core.core.ecs.component import (
    HealthComponent,
    InventoryComponent,
    PositionComponent,
    TimeBudgetComponent,
)
from agent_core.policy.state_encoder_interface import StateEncoderInterface
from agent_engine.utils.math_utils import safe_divide


class SoulSimStateEncoder(StateEncoderInterface):
    """Encodes the soul_sim world state into a numerical vector."""

    def encode_state(
        self, simulation_state: Any, entity_id: str, config: Dict[str, Any], target_entity_id: Optional[str] = None
    ) -> np.ndarray:
        components = simulation_state.entities.get(entity_id, {})
        pos, time, health, inv = (
            components.get(c) for c in [PositionComponent, TimeBudgetComponent, HealthComponent, InventoryComponent]
        )
        if not all([pos, time, health, inv]):
            # Return a zero vector if any core component is missing
            return np.zeros(16, dtype=np.float32)

        env = simulation_state.environment

        # Normalize features to be roughly between 0 and 1
        self_features = [
            safe_divide(pos.position[0], env.height),
            safe_divide(pos.position[1], env.width),
            safe_divide(health.current_health, health.initial_health),
            safe_divide(time.current_time_budget, time.initial_time_budget),
            min(1.0, safe_divide(inv.current_resources, 200.0)), # Cap resources feature
        ]

        # NOTE: These are placeholders. In a real implementation, you would
        # fill these with data about nearby entities, resources, etc.
        other_features = [0.0] * 4
        resource_features = [0.0] * 4
        social_features = [0.0] * 3

        # Ensure final vector is the correct shape and type
        feature_vector = np.array(self_features + other_features + resource_features + social_features, dtype=np.float32)

        # This is a safeguard against generating vectors of the wrong size
        expected_size = config.get("learning", {}).get("q_learning", {}).get("state_feature_dim", 16)
        if feature_vector.size != expected_size:
            # Pad with zeros or truncate if size is wrong
            final_vector = np.zeros(expected_size, dtype=np.float32)
            size_to_copy = min(feature_vector.size, expected_size)
            final_vector[:size_to_copy] = feature_vector[:size_to_copy]
            return final_vector

        return feature_vector
