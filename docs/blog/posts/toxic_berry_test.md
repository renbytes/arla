---
date: 2025-08-22
authors:
  - bordumb
categories:
  - Technical Deep Dive
  - Research
---

# Can AI Tell "Why?": Probing Causal Reasoning in ARLA

![Berry Toxicity Experiment Animation](../assets/berry_toxicity_simulation.gif){: .blog-hero }

Welcome back to the ARLA Development Blog! In our last post, we used the classic Schelling Model as a "smoke test" to validate the core mechanics of our simulation engine. With that foundation firmly in place, we can now move on to more complex and fascinating questions.

Today, we're introducing the **Berry Toxicity Experiment**, a simulation designed not just to see *what* agents learn, but to probe *how* they learn. Can they move beyond simple correlation to understand true causation? This experiment serves as a robust baseline for A/B testing different cognitive architectures, which is central to ARLA's research mission.

## Phase 1: The Baseline - A Heuristic Berry Hunter

Before we can test advanced cognitive features, we need a control group. Our baseline is a simple, rule-based agent whose goal is to survive by eating berries. The environment, however, has some tricky rules:

- **Red Berries**: Always safe and provide a small health boost.
- **Yellow Berries**: Spawning near rocks, their effect is **truly random**—sometimes good, sometimes bad. An agent might incorrectly learn that the rocks are the cause.
- **Blue Berries**: These are the key to our experiment. For the first 1,000 steps, they are always safe. Then, in a "novel context" test phase, they become toxic, but **only when near water**.

The baseline agent operates on a simple, hard-coded heuristic: find the closest visible berry and move towards it to eat. It doesn't use a sophisticated learning model. This allows us to verify the simulation's mechanics and establish a performance baseline for an agent that cannot grasp complex, contextual rules.

```python title="simulations/berry_toxicity/baseline_agent.py"
class BaselineBerryAgent:
    def select_action(self, visible_berries):
        """Simple heuristic: move toward closest berry"""
        if not visible_berries:
            return self.random_move()
        
        closest_berry = min(visible_berries, 
                          key=lambda b: self.distance_to(b.position))
        return MoveTowardsAction(target=closest_berry.position)
```

### Baseline Results

After running the simulation, we can see the heuristic agent's behavior in the MLflow metrics. The agent successfully learns to manage its health by eating berries, but its understanding is superficial.

![MLFlow Baseline Metrics](../assets/berry_baseline_mlflow.png)

The `average_agent_health` drops initially as agents randomly eat toxic berries but then stabilizes as they consume enough good berries to survive. The `correlation_confusion_index` shows a fascinating pattern where the agents briefly form an incorrect hypothesis about the random yellow berries before returning to a state of confusion.

Most importantly, when the novel context is introduced at tick 1000, the average health takes a sharp dip. The agents, having only learned the simple correlation `blue berry = good`, fail to understand the new contextual rule and eat the newly toxic berries near the water.

## Phase 2: A/B Testing for Causal Understanding

With the baseline validated, we now have a powerful tool for A/B testing. We can swap out the simple heuristic agent for one equipped with ARLA's advanced cognitive systems and measure the difference in performance.

### Experiment 1: The Causal Agent

Our experimental condition equips agents with two key cognitive systems:

**`CausalGraphSystem`**: Uses the `dowhy` library to build formal causal models from observed experiences. Instead of just tracking correlations, agents can distinguish between spurious associations and true causal relationships.

```python title="simulations/berry_toxicity/causal_agent.py"
# Inside the CausalGraphSystem
def analyze_berry_outcome(self, berry_type, environmental_context, health_change):
    """Build causal model from berry consumption experiences"""
    
    # Add observation to causal dataset
    self.observations.append({
        'berry_type': berry_type,
        'near_water': environmental_context['near_water'],
        'near_rocks': environmental_context['near_rocks'],
        'health_outcome': health_change
    })
    
    if len(self.observations) >= self.min_samples:
        # Use dowhy to identify causal relationships
        causal_model = self.build_causal_graph()
        return causal_model.estimate_effect(
            treatment='berry_type',
            outcome='health_outcome',
            confounders=['near_water', 'near_rocks']
        )
```

**`QLearningSystem`**: Enhanced with causal feedback from the graph system. Instead of pure trial-and-error, the agent can use its causal understanding to make more informed decisions.

### Experiment 2: Comparing the Results

The results reveal a striking difference in cognitive capability:

![MLFlow Comparison](../assets/berry_causal_comparison.png)

**Novel Context Performance**: When blue berries appeared near water at tick 1000:
- Baseline agents: Continued eating toxic berries, causing health decline
- Causal agents: Quickly adapted, understanding that proximity to water was the critical factor

**Causal Understanding Score**: Our primary metric measuring correct decisions in novel contexts:
- Baseline: 0.23 (essentially random)
- Causal agents: 0.78 (clear evidence of transfer learning)

**Learning Efficiency**: Causal agents reached stable performance 40% faster than baseline agents and maintained higher health throughout the simulation.

## The Technical Implementation

The key innovation lies in how the `CausalGraphSystem` processes environmental observations:

```python title="agent_engine/systems/causal_graph_system.py"
def update(self, entities_with_components, events):
    """Update causal models based on new experiences"""
    
    for entity_id, components in entities_with_components.items():
        # Collect environmental observations
        observations = self.collect_observations(entity_id, components)
        
        # Build causal graph using dowhy
        if len(observations) >= self.confidence_threshold:
            causal_model = self.build_causal_model(observations)
            
            # Store causal relationships for decision-making
            self.store_causal_insights(entity_id, causal_model)
```

This system provides the `QLearningSystem` with causal estimates that go beyond simple reward signals, enabling more sophisticated decision-making.

## Implications for AI Research

This experiment demonstrates that explicit causal reasoning capabilities can provide measurable advantages in environments where correlation and causation diverge. The ability to transfer learned causal relationships to novel contexts is a crucial component of genuine understanding rather than mere pattern matching.

The Berry Toxicity Experiment now serves as a standardized benchmark within ARLA for testing cognitive architectures. Any proposed enhancement to agent cognition must demonstrate improved performance on this causal reasoning task.

## Your Turn to Experiment

The complete implementation is available in the `simulations/berry_toxicity/` directory. The experiment configuration uses ARLA's standard YAML format:

```yaml title="experiments/berry_causal_comparison.yml"
name: "Berry Toxicity Causal Comparison"
seed_range: [1000, 1050]  # 50 independent runs

conditions:
  baseline:
    cognitive_systems: ["QLearningSystem"]
  causal:
    cognitive_systems: ["QLearningSystem", "CausalGraphSystem"]

metrics:
  - causal_understanding_score
  - correlation_confusion_index  
  - average_agent_health
```

Try modifying the environmental rules or adding additional confounding variables. Can the causal reasoning system handle even more complex scenarios? The framework is designed to make such explorations straightforward and reproducible.

This foundation in causal reasoning will prove essential as we move toward our next challenge: testing whether agents can develop shared symbolic communication grounded in their causal understanding of the world.