# Formal Causal Reasoning in the ARLA Framework

This document outlines the formal causal reasoning engine implemented in the ARLA framework, which replaces the previous correlation-based CausalGraphSystem.

## 1. The "Why": Moving Beyond Correlation

The original CausalGraphSystem was effective at identifying which events or states were correlated with positive or negative outcomes. However, correlation is not causation. An agent might learn that being near a "Danger" sign is associated with a negative reward, but it couldn't distinguish between two possibilities:

1. The sign itself is harmless, but it happens to be located near a cliff that causes the negative reward (confounding).
2. The sign is a trap that directly causes the negative reward.

To build more intelligent and robust agents, we need a system that can understand true cause-and-effect relationships. This allows an agent to learn the real consequences of its actions, leading to better decision-making.

## 2. The "How": Integration with DoWhy

We have integrated the `dowhy` library, a powerful tool for causal inference, to build and analyze formal causal models for each agent.

### Data Collection

- The `ActionSystem` now attaches a unique `event_id` to every action's outcome.
- The `CausalGraphSystem` listens for `action_executed` events. For each event, it logs a structured record `(event_id, pre-action_state, action, outcome)` to a list within the agent's `MemoryComponent`.

### Causal Model Construction

- Periodically (e.g., every 50 ticks), the `CausalGraphSystem` takes the collected data from an agent's `MemoryComponent` and constructs a `dowhy.CausalModel`.
- This model is built using a predefined causal graph, which represents our domain knowledge about how variables in the simulation are related. For example, we assume that an agent's state influences its choice of action, and that both the state and the action can influence the outcome.
- The constructed `CausalModel` is then stored back in the agent's `MemoryComponent`.

### Estimating Causal Effects (Do-Calculus)

- The `CausalGraphSystem` exposes a new method: `estimate_causal_effect(agent_id, treatment_value)`.
- This method uses the agent's stored `CausalModel` to perform a causal intervention based on Judea Pearl's do-calculus. It answers the question: "What would the average effect on the outcome be if the agent were forced to take this action, regardless of the state it was in?"
- This is different from simply averaging the observed rewards for that action, as it controls for confounding variables (e.g., "Did the agent succeed because it chose to fight, or because it only chose to fight when it was already in a strong state?").

### Formal Counterfactual Reasoning

- The `generate_counterfactual` function now uses the agent's `CausalModel` to ask "what if?" questions about specific past events.
- Using the `event_id`, it finds the exact state of the world during a past action and uses the model's `whatif()` method to predict what the outcome would have been had the agent taken a different action.

## 3. The "What": How to Use the Causal Engine

The new causal reasoning capabilities are consumed by other systems to improve the agent's intelligence.

### For Learning Systems (QLearningSystem)

- The `QLearningSystem` now calls `estimate_causal_effect` after an action is performed.
- It then blends the raw, observed reward with the causally-estimated reward. This creates a more robust learning signal that is less susceptible to spurious correlations. The agent learns based on what its actions truly cause.

### For Reflection Systems (counterfactual.py)

- To generate a counterfactual, a system can now call `generate_counterfactual` and pass in a specific past episode and an alternative action to consider.
- The function will return a `CounterfactualEpisode` object containing a mathematically grounded prediction, which can be used to generate new insights, goals, or beliefs for the agent.

### For Model Validation

- The `CausalModelValidator` class in `validation.py` can be used to test the robustness of an agent's learned causal model.
- It runs several refutation tests (e.g., adding a random common cause, using a placebo treatment) and produces a confidence score.
- This score is stored in the `ValidationComponent` and can be used by the agent to understand how much it should trust its own causal beliefs.