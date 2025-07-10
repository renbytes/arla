# actions/action_definition.py

from typing import TYPE_CHECKING, Any, Dict, List

from agent_core.agents.actions.action_registry import action_registry
from agent_core.agents.actions.base_action import Intent


def create_standard_feature_vector(
    action_id: str,
    intent: Intent,
    base_cost: float,
    params: Dict[str, Any],
    param_feature_map: Dict[str, Any],
) -> List[float]:
    """Helper function to create a standardized action feature vector."""
    action_ids = action_registry.action_ids
    action_one_hot = [1.0 if action_id == i else 0.0 for i in action_ids]
    intents = list(Intent)
    intent_one_hot = [1.0 if i == intent else 0.0 for i in intents]
    time_cost_feature = [base_cost / 25.0]
    param_features = [0.0] * 5
    for param_name, mapping in param_feature_map.items():
        if param_name in params:
            idx, value, normalizer = mapping
            param_features[idx] = float(value) / float(normalizer) if normalizer != 0 else 0.0
    return action_one_hot + intent_one_hot + time_cost_feature + param_features
