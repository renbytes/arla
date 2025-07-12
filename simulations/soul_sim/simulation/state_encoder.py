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
from simulations.soul_sim.components import CombatComponent, ResourceComponent


class SoulSimStateEncoder(StateEncoderInterface):
    """Encodes the soul_sim world state into a numerical vector."""

    def encode_state(
        self, simulation_state: Any, entity_id: str, config: Dict[str, Any], target_entity_id: Optional[str] = None
    ) -> np.ndarray:
        """Creates a feature vector from soul-sim's specific components."""
        components = simulation_state.entities.get(entity_id, {})
        pos_comp, time_comp, health_comp, inv_comp = (
            components.get(c) for c in [PositionComponent, TimeBudgetComponent, HealthComponent, InventoryComponent]
        )
        if not all([pos_comp, time_comp, health_comp, inv_comp]):
            return np.zeros(16, dtype=np.float32)

        env = simulation_state.environment
        max_dist = np.sqrt(env.width**2 + env.height**2)

        # --- Self Features ---
        self_features = [
            safe_divide(pos_comp.position[0], env.height),
            safe_divide(pos_comp.position[1], env.width),
            health_comp.normalized,
            safe_divide(time_comp.current_time_budget, time_comp.initial_time_budget),
            min(1.0, safe_divide(inv_comp.current_resources, 200.0)),
        ]

        # --- Nearest Resource Features ---
        resource_features = [0.0] * 4  # [dist, health, type1, type2]
        closest_res_dist = float('inf')
        all_resources = simulation_state.get_entities_with_components([ResourceComponent, PositionComponent])
        for res_id, res_comps in all_resources.items():
            res_pos_comp = res_comps.get(PositionComponent)
            dist = env.distance(pos_comp.position, res_pos_comp.position)
            if dist < closest_res_dist:
                closest_res_dist = dist
                res_comp = res_comps.get(ResourceComponent)
                resource_features = [
                    dist / max_dist,
                    safe_divide(res_comp.current_health, res_comp.initial_health),
                    1.0 if "DOUBLE" in res_comp.type else 0.0,
                    1.0 if "TRIPLE" in res_comp.type else 0.0,
                ]

        # --- Nearest Agent Features ---
        other_agent_features = [0.0] * 4 # [dist, health, attack, is_target]
        closest_agent_dist = float('inf')
        all_agents = simulation_state.get_entities_with_components([TimeBudgetComponent, PositionComponent])
        for other_id, other_comps in all_agents.items():
            if other_id == entity_id:
                continue
            other_pos_comp = other_comps.get(PositionComponent)
            dist = env.distance(pos_comp.position, other_pos_comp.position)
            if dist < closest_agent_dist:
                closest_agent_dist = dist
                other_health = other_comps.get(HealthComponent)
                other_combat = other_comps.get(CombatComponent)
                other_agent_features = [
                    dist / max_dist,
                    other_health.normalized if other_health else 0.0,
                    safe_divide(other_combat.attack_power, 20.0) if other_combat else 0.0,
                    1.0 if other_id == target_entity_id else 0.0
                ]

        # --- Social/Misc Features (can be expanded) ---
        social_features = [0.0] * 3

        # --- Combine and Pad/Truncate ---
        feature_vector = np.array(
            self_features + resource_features + other_agent_features + social_features, dtype=np.float32
        )

        expected_size = config.get("learning", {}).get("q_learning", {}).get("state_feature_dim", 16)
        if feature_vector.size != expected_size:
            final_vector = np.zeros(expected_size, dtype=np.float32)
            size_to_copy = min(feature_vector.size, expected_size)
            final_vector[:size_to_copy] = feature_vector[:size_to_copy]
            return final_vector

        return feature_vector
