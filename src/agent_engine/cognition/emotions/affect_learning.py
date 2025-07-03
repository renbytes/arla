# src/cognition/emotions/affect_learning.py

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from sklearn.cluster import KMeans

from agent_core.agents.actions.action_registry import action_registry
from .affect_base import AffectiveExperience


def name_experience_cluster(
    experiences: List[AffectiveExperience],
    cognitive_scaffold: Any,
    agent_id: str,
    current_tick: int,
    prompt_template: str,
    purpose: str,
) -> str:
    """
    Generic helper to call an LLM to name a cluster of experiences using a provided prompt template.
    """
    prompt_samples = []
    action_ids = action_registry.action_ids
    for exp in experiences:
        action_name = "UNKNOWN_ACTION"
        if np.any(exp.action_type_one_hot):
            action_index = np.argmax(exp.action_type_one_hot)
            if action_index < len(action_ids):
                action_instance_type = action_registry.get_action(action_ids[action_index])
                action_instance = action_instance_type()
                action_name = action_instance.name

        prompt_samples.append(
            f"(Action: {action_name}, Reward: {exp.outcome_reward:.1f}, Valence: {exp.valence:.2f}, Arousal: {exp.arousal:.2f})"
        )

    # Use an f-string to inject the samples into the template
    final_prompt = prompt_template.format(summaries="; ".join(prompt_samples))

    try:
        # The query method is guaranteed to return a string, token usage, and cost.
        name: str = (
            cognitive_scaffold.query(
                agent_id=agent_id,
                purpose=purpose,
                prompt=final_prompt,
                current_tick=current_tick,
            )[0]
            .strip()
            .replace('"', "")
        )
        return name if name else "unnamed"
    except Exception as e:
        print(f"Error naming cluster for purpose '{purpose}': {e}")
        return "unnamed"


def _cluster_experiences(affect_comp: Any, config: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Performs KMeans clustering on the affective experience buffer."""
    learning_memory_config = config.get("learning", {}).get("memory", {})
    data_vectors_list = [exp.vector for exp in affect_comp.affective_experience_buffer]

    if not data_vectors_list:  # Handle empty list case
        return None, None

    data_vectors = np.array(data_vectors_list)

    num_clusters_raw = len(data_vectors) // (learning_memory_config.get("emotion_cluster_min_data", 50) // 4)
    num_clusters = max(2, num_clusters_raw)
    num_clusters = min(num_clusters, 5)

    if len(data_vectors) < num_clusters:
        return None, None

    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(data_vectors)
        return kmeans.labels_, kmeans.cluster_centers_
    except ValueError as e:
        print(f"Error during KMeans: {e}. Skipping emotion clustering.")
        return None, None


def discover_emotions(
    affect_comp: Any,
    cognitive_scaffold: Any,
    agent_id: str,
    current_tick: int,
    config: Dict[str, Any],
) -> None:
    """
    Performs unsupervised clustering on the agent's affective experience buffer
    to discover and name emotion categories.
    """
    learning_memory_config = config.get("learning", {}).get("memory", {})

    if len(affect_comp.affective_experience_buffer) < learning_memory_config["emotion_cluster_min_data"]:
        return

    labels, centroids = _cluster_experiences(affect_comp, config)
    if labels is None or centroids is None:
        return

    # Define the specific prompt for naming EMOTIONS
    emotion_prompt_template = """I experienced a series of internal states and actions like these:
            {summaries}. What is a concise, single-word emotion or
            feeling that best describes this cluster of experiences?
            Only return the word, e.g., 'joy', 'frustration', 'calm'."""

    new_clusters = {}
    for i, centroid in enumerate(centroids):
        cluster_indices = np.where(labels == i)[0]
        if not cluster_indices.size:
            continue

        sample_experiences = [
            affect_comp.affective_experience_buffer[j]
            for j in np.random.choice(cluster_indices, min(5, len(cluster_indices)), replace=False)
        ]

        # Pass the new context down to the helper function
        emotion_name = name_experience_cluster(
            experiences=sample_experiences,
            cognitive_scaffold=cognitive_scaffold,
            agent_id=agent_id,
            current_tick=current_tick,
            prompt_template=emotion_prompt_template,
            purpose="emotion_cluster_naming",
        )

        original_name = emotion_name
        k = 1
        while emotion_name in new_clusters:
            emotion_name = f"{original_name}_{k}"
            k += 1

        new_clusters[emotion_name] = {
            "centroid": centroid,
            "samples": [exp.to_dict() for exp in sample_experiences],
            "count": len(cluster_indices),
        }

    affect_comp.learned_emotion_clusters = new_clusters
    affect_comp.affective_experience_buffer.clear()
    print(f"Discovered {len(new_clusters)} emotion clusters for agent {agent_id}: {list(new_clusters.keys())}")


def get_emotion_from_affect(
    valence: float,
    arousal: float,
    prediction_delta_magnitude: float,
    predictive_delta_smooth: float,
    health_norm: float,
    time_norm: float,
    res_norm: float,
    action_type_one_hot: np.ndarray,
    outcome_reward: float,
    prediction_error: float,
    is_positive_outcome: bool,
    learned_emotion_clusters: Dict[str, Any],
) -> str:
    """
    Determines the current emotion label based on learned clusters.
    """
    if not learned_emotion_clusters:
        return "unknown_emotion"

    current_experience_vector = AffectiveExperience(
        valence,
        arousal,
        prediction_delta_magnitude,
        predictive_delta_smooth,
        health_norm,
        time_norm,
        res_norm,
        action_type_one_hot,
        outcome_reward,
        prediction_error,
        is_positive_outcome,
    ).vector

    min_dist: float = float("inf")
    closest_emotion = "unknown_emotion"
    for name, cluster_data in learned_emotion_clusters.items():
        centroid = cast(np.ndarray, cluster_data["centroid"])
        dist = np.linalg.norm(current_experience_vector - centroid)
        if dist < min_dist:
            min_dist = float(dist)
            closest_emotion = name
    return closest_emotion
