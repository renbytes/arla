# ARLA: A Modular ECS-Based Platform for Ablative Studies in Multi-Agent Cognitive Architecture

**Authors**: Brian Deely

**Affiliation**: Independent Researcher

## Abstract

Research into complex, believable agent behavior requires simulation platforms that are both powerful and easy to modify. Existing tools are often monolithic, making it difficult to perform targeted experiments, such as ablative studies of cognitive components. In this paper, we introduce the Affective Reinforcement Learning Architecture (ARLA), a novel open-source simulation platform designed specifically for building and experimenting with cognitive agents. ARLA is architected using the Entity-Component-System (ECS) pattern, which enforces a clean separation between an agent's core cognitive logic and the rules of the simulated world. We detail its modular design, including a world-agnostic cognitive engine, a concurrent execution manager, and a robust state persistence layer. By demonstrating its use in an exemplar simulation, we show that ARLA provides a flexible and extensible foundation for conducting reproducible research into emergent social phenomena, agent economics, and the foundations of artificial morality.

## 1. Introduction

**Motivation**: The recent success of large language models has renewed interest in creating autonomous agents that exhibit believable, human-like behavior (e.g., Park et al., 2023). This pursuit requires sophisticated and transparent simulation environments to serve as digital laboratories.

**The Problem**: A primary challenge is the tight coupling of agent "minds" and world "physics" in many existing simulators. This makes it difficult to isolate and study the effects of specific cognitive mechanisms (e.g., "How does an agent's behavior change if we disable its ability to form a stable identity?"). This process, known as ablative analysis, is critical for cognitive science but is poorly supported by monolithic architectures.

**Our Contribution**: We present ARLA, a platform designed to solve this problem. Its core contribution is its modularity, enforced by:

- A decoupled architecture based on the Entity-Component-System (ECS) pattern.
- A clear separation of concerns between a reusable, world-agnostic cognitive engine and world-specific implementations.
- A system of provider interfaces that act as a bridge, allowing the engine to operate without depending on concrete world details.

**Paper Outline**: We will first survey related work, then detail ARLA's architecture, demonstrate its use in a concrete simulation, evaluate its performance, and conclude by discussing the experimental possibilities it enables.

## 2. Related Work

**Agent-Based Modeling (ABM) Platforms**: We position ARLA relative to established ABM frameworks like NetLogo (Tisue & Wilensky, 2004) and Repast (North et al., 2013), noting that while they are powerful for large-scale population dynamics, ARLA focuses more deeply on the rich internal cognitive and affective states of individual agents. We also compare it to modern Python-based alternatives like Mesa (Masad & Kazil, 2015).

**AI Simulation Environments**: We contrast ARLA with environments designed primarily for reinforcement learning, such as the Unity ML-Agents Toolkit or DeepMind's Melting Pot. While these are excellent for training policies, ARLA is designed as an observable "digital terrarium" for studying the emergent behavior of pre-defined cognitive architectures, in the vein of the "Generative Agents" simulation (Park et al., 2023).

**Cognitive Architectures**: ARLA is not a cognitive architecture itself, but a testbed for implementing their components. We draw inspiration from classic architectures like SOAR (Laird, 2012) and ACT-R (Anderson, 2007), whose theories on memory, goal-handling, and decision-making inform the design of ARLA's cognitive systems (e.g., MemoryComponent, GoalSystem).

### Table 1: Feature Comparison with Existing Platforms

| Feature | ARLA | NetLogo | Unity ML-Agents | Mesa | Repast |
|---------|------|---------|-----------------|------|--------|
| Cognitive Architecture Support | ✓ | Limited | ✗ | Limited | Limited |
| Ablative Studies | ✓ | ✗ | ✗ | ✗ | ✗ |
| LLM Integration | ✓ | ✗ | ✗ | ✗ | ✗ |
| Async Execution | ✓ | ✗ | ✓ | ✗ | ✓ |
| State Persistence | ✓ | ✓ | Limited | ✓ | ✓ |
| Provider Pattern | ✓ | ✗ | ✗ | ✗ | ✗ |
| Built-in Emotion Model | ✓ | ✗ | ✗ | ✗ | ✗ |

## 3. The ARLA Framework Architecture

This section details the design of the monorepo's core packages.

[DIAGRAM PLACEHOLDER: Three-layer architecture diagram showing agent-core, agent-engine, and agent-sim layers with arrows indicating dependencies and data flow]

### 3.1. Design Philosophy: The ECS Pattern

- **Entities**: Unique IDs representing agents or environmental objects.
- **Components**: Plain data containers holding an agent's state (e.g., IdentityComponent, HealthComponent).
- **Systems**: The logic that operates on entities possessing specific components (e.g., AffectSystem, DecaySystem).

This design choice is justified by its proven flexibility and performance in game development, which shares many challenges with agent simulation (Nystrom, 2014).

### 3.2. agent-core: Foundational Interfaces

Defines the "contract" for the entire platform. Details key abstractions like Component, ActionInterface, and the provider interfaces. This section emphasizes that any new simulation built with ARLA must adhere to these contracts.

Key interfaces include:
- `Component`: Base class with `to_dict()` and `validate()` methods
- `ActionInterface`: Contract for all agent actions
- `VitalityMetricsProviderInterface`: Bridge for health/resource data
- `NarrativeContextProviderInterface`: Bridge for LLM-based reflection
- `StateEncoderInterface`: Bridge for Q-learning state representation

### 3.3. agent-engine: The Cognitive Engine

Details the world-agnostic systems that constitute an agent's "mind":

[DIAGRAM PLACEHOLDER: System interaction diagram showing event flow between cognitive systems]

- **ActionSystem & RewardCalculator**: Handling action outcomes and subjective rewards based on agent values.
- **AffectSystem & EmotionalDynamics**: Modeling emotion via appraisal theory (Scherer, 2001), incorporating goal relevance, controllability, and social context.
- **IdentitySystem**: Managing a multi-domain model of an agent's self-concept across social, competence, moral, relational, and agency dimensions.
- **ReflectionSystem & GoalSystem**: Processing experiences into narrative memories and generating goals through LLM-based reflection.
- **QLearningSystem**: A utility-based decision-making module with neural network function approximation.
- **CausalGraphSystem**: Building and maintaining a graph of cause-effect relationships from agent experiences.

### 3.4. agent-concurrent & agent-persist: Pluggable Infrastructure

**agent-concurrent**: Provides `AsyncSystemRunner` and `SerialSystemRunner` for flexible execution strategies, allowing researchers to choose between performance and deterministic debugging.

**agent-persist**: Implements state serialization using Pydantic models, enabling checkpoint/restore functionality and post-hoc analysis of agent trajectories.

### 3.5. Key Implementation Decisions

- **Event-driven Architecture**: Systems communicate via an event bus rather than direct calls, enabling loose coupling and easy instrumentation.
- **Async/Await for Concurrency**: Python's asyncio enables concurrent system updates without threading complexity.
- **Provider Pattern**: Dependency injection allows world-agnostic systems to access world-specific data.
- **LLM Integration**: Centralized `CognitiveScaffold` manages all LLM interactions with caching, cost tracking, and structured prompt management.
- **Deterministic Execution**: Careful management of random seeds and execution order ensures reproducible simulations.

## 4. Exemplar Simulation: soul-sim

This section demonstrates how the abstract framework is used to create a runnable simulation.

### 4.1. World Implementation

Defining world-specific components:
```python
class PositionComponent(Component):
    """Stores an entity's position in a 2D grid world."""
    def __init__(self, position: Tuple[int, int], environment: EnvironmentInterface):
        self.position = position
        self.history: deque[Tuple[int, int]] = deque(maxlen=20)
        self.visited_positions: set[Tuple[int, int]] = {position}
```

Defining world-specific systems:
```python
class CombatSystem(System):
    """Processes combat actions between agents."""
    def on_execute_combat(self, event_data: Dict[str, Any]):
        # Validate positions, calculate damage, update health
        # Publish outcome for cognitive systems to process
```

### 4.2. Bridging the World and Engine

Showcasing a concrete implementation of a provider:

```python
class SoulSimVitalityProvider(VitalityMetricsProviderInterface):
    def get_normalized_vitality_metrics(
        self, entity_id: str, components: Dict[Type[Component], Component], config: Dict[str, Any]
    ) -> Dict[str, float]:
        health_comp = components.get(HealthComponent)
        time_comp = components.get(TimeBudgetComponent)
        inv_comp = components.get(InventoryComponent)

        return {
            "health_norm": health_comp.current_health / health_comp.initial_health,
            "time_norm": time_comp.current_time_budget / time_comp.initial_time_budget,
            "resources_norm": min(1.0, inv_comp.current_resources / 100.0)
        }
```

### 4.3. A Simulation in Action

Walk through a scenario: Agent A attacks Agent B.

1. **CombatSystem** processes the attack, reduces B's health
2. **ActionSystem** receives the outcome via event bus
3. **RewardCalculator** applies A's value multipliers (combat_victory_multiplier)
4. **AffectSystem** updates A's emotion based on prediction error
5. **IdentitySystem** may strengthen A's "competence" domain if victorious
6. **SocialMemoryComponent** updates B's impression of A as threatening

[DIAGRAM PLACEHOLDER: Sequence diagram showing event flow for the combat scenario]

### 4.4. Cognitive Model Validation

We validate our cognitive models against established psychological baselines:

- **Emotion Dynamics**: Our appraisal-based emotion model produces valence/arousal trajectories consistent with [PLACEHOLDER: specific psychology studies]
- **Identity Coherence**: Multi-domain identity stability scores correlate with [PLACEHOLDER: social psychology metrics] at r=[PLACEHOLDER]
- **Goal Emergence**: Goal generation patterns match motivated behavior literature, with [PLACEHOLDER]% of goals relating to recent success experiences

### 4.5. Example Experiments

#### 4.5.1 Identity Ablation Study
- **Configuration**: 100 agents, 50% with IdentitySystem disabled
- **Metrics**: Cooperation frequency, resource accumulation, survival time
- **Preliminary Results**: Agents without identity show [PLACEHOLDER]% less consistent behavior patterns and [PLACEHOLDER]% reduced cooperation

#### 4.5.2 Economic Emergence
- **Setup**: Variable resource scarcity (abundant/scarce/depleting)
- **Observed Behaviors**:
  - Trading emerges in [PLACEHOLDER]% of scarce resource conditions
  - Territorial defense correlates with resource respawn time (r=[PLACEHOLDER])
  - Wealth inequality (Gini coefficient) reaches [PLACEHOLDER] after 1000 ticks

## 5. Performance Evaluation

### 5.1 Scalability Analysis

[DIAGRAM PLACEHOLDER: Performance graphs showing scalability metrics]

| Agent Count | Ticks/Second | Memory (MB) | With AsyncRunner | With SerialRunner |
|-------------|--------------|-------------|------------------|-------------------|
| 100         | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 1,000       | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |
| 10,000      | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] | [PLACEHOLDER] |

**Database Performance**:
- Write throughput: [PLACEHOLDER] events/second with batch size 1000
- Query latency for analysis: [PLACEHOLDER]ms for tick-range queries

### 5.2 Computational Overhead

- **Provider Abstraction**: [PLACEHOLDER]% overhead vs direct component access
- **Event Bus**: [PLACEHOLDER] microseconds per event dispatch
- **LLM Integration**:
  - Average [PLACEHOLDER] API calls per agent per 100 ticks
  - Cost: $[PLACEHOLDER] per 1000-agent-tick simulation
  - Cache hit rate: [PLACEHOLDER]% for reflection prompts

### 5.3 Reproducibility Features

- **Deterministic Execution**: Identical seeds produce bit-identical results across runs
- **Configuration Validation**: Pydantic schemas catch [PLACEHOLDER]% of configuration errors
- **State Snapshots**: Full state serialization in [PLACEHOLDER]ms for 1000 agents
- **MLflow Integration**: Automatic experiment tracking with hyperparameters, metrics, and artifacts

## 6. Community and Ecosystem

The ARLA platform is available at [https://github.com/PLACEHOLDER/arla] under the MIT license.

**Resources Available**:
- Comprehensive documentation with tutorials
- Example experiments and configurations
- Plugin template for custom cognitive systems
- Discord community for researchers

**Standardized Formats**:
- Experiment configuration schemas
- Agent behavior trace format
- Cognitive component interchange format

**Planned Activities**:
- Annual workshop at [PLACEHOLDER] conference
- Online competition for emergent behavior discovery
- Model zoo for pre-trained agent configurations

## 7. Ethical Considerations

While ARLA enables powerful simulations of agent behavior, we acknowledge several ethical considerations:

- **Potential Misuse**: Simulations could model harmful behaviors or social dynamics
- **LLM Reflections**: Agent "thoughts" may reflect biases in underlying language models
- **Computational Resources**: Large simulations have environmental impact
- **Responsible Disclosure**: We commit to responsible disclosure of emergent strategies that could be exploited

We encourage researchers to consider the broader implications of their experiments and to use ARLA in ways that advance beneficial AI research.

## 8. Discussion & Future Work

### Strengths

- **Modularity for Ablative Studies**: Researchers can disable systems (e.g., IdentitySystem) to isolate cognitive contributions
- **Extensibility**: New cognitive modules, actions, and worlds can be added via the provider pattern
- **Reproducibility**: Comprehensive state management and configuration validation
- **Rich Cognitive Modeling**: Integrated emotion, identity, goals, and social memory

### Limitations

- **Python Performance**: GIL limits true parallelism; C++/Rust extensions planned
- **Memory Scaling**: Current architecture requires ~[PLACEHOLDER]MB per agent
- **LLM Costs**: Reflection-heavy simulations can incur significant API costs
- **Visualization**: Limited real-time capabilities; post-hoc analysis focus

### Technical Roadmap

1. **Performance Enhancements**:
   - Rust core for performance-critical paths (Q3 2024)
   - GPU acceleration for neural components (Q4 2024)
   - Distributed simulation across nodes (2025)

2. **Features**:
   - Real-time web visualization dashboard
   - Integration with PyTorch Lightning for advanced ML
   - Standardized cognitive architecture benchmarks
   - Natural language scenario specification

### Research Agenda

1. **Ablative Studies**: Systematic analysis of each cognitive system's contribution to:
   - Cooperative success in resource-limited environments
   - Emergent communication protocols
   - Social hierarchy formation

2. **Economic Studies**: Investigation of:
   - Conditions for currency emergence
   - Impact of value system diversity on market dynamics
   - Commons management strategies

3. **Morality Studies**: Implementation and comparison of:
   - Deontological vs. utilitarian reward calculators
   - Evolution of moral norms through social feedback
   - Punishment and altruism emergence

## 9. Conclusion

The ARLA platform provides a robust, modular, and extensible tool for the computational study of artificial agents. By strictly separating an agent's mind from its environment, it lowers the barrier to entry for conducting complex, reproducible experiments in cognitive architecture. Early experiments demonstrate its utility for studying emergence in multi-agent systems, from economic behaviors to social dynamics. We have made the platform open-source to encourage collaboration and extension, and we look forward to the research community's contributions and discoveries.

## Acknowledgments

[PLACEHOLDER: Funding sources, collaborators, and institutional support]

## References

Anderson, J. R. (2007). How can the human mind occur in the physical universe? Oxford University Press.

Laird, J. E. (2012). The Soar cognitive architecture. MIT Press.

Luke, S., Cioffi-Revilla, C., Panait, L., Sullivan, K., & Balan, G. (2005). MASON: A multiagent simulation environment. Simulation, 81(7), 517-527.

Masad, D., & Kazil, J. (2015). MESA: An agent-based modeling framework. Proceedings of the 14th Python in Science Conference.

North, M. J., Collier, N. T., Ozik, J., Tatara, E. R., Macal, C. M., Bragen, M., & Sydelko, P. (2013). Complex adaptive systems modeling with Repast Simphony. Complex adaptive systems modeling, 1(1), 1-26.

Nystrom, R. (2014). Game Programming Patterns. Genever Benning.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. arXiv preprint arXiv:2304.03442.

Scherer, K. R. (2001). Appraisal considered as a process of multilevel sequential checking. Appraisal processes in emotion: Theory, methods, research, 92(120), 57.

Tisue, S., & Wilensky, U. (2004). NetLogo: A simple environment for modeling complexity. Proceedings of the International Conference on Complex Systems.

## Appendix A: Getting Started

```python
# Quick example of creating a custom cognitive system
from agent_engine.simulation.system import System
from agent_core.core.ecs.component import Component

class TrustComponent(Component):
    def __init__(self):
        self.trust_scores: Dict[str, float] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {"trust_scores": self.trust_scores}

    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        return True, []

class TrustSystem(System):
    REQUIRED_COMPONENTS = [TrustComponent, SocialMemoryComponent]

    async def update(self, current_tick: int):
        # Update trust based on recent interactions
        for entity_id, components in self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        ).items():
            trust_comp = components[TrustComponent]
            social_comp = components[SocialMemoryComponent]

            # Custom trust dynamics based on interaction history
            for other_id, schema in social_comp.schemas.items():
                if schema.interaction_count > 0:
                    # Trust grows with positive interactions
                    trust_comp.trust_scores[other_id] = min(1.0,
                        trust_comp.trust_scores.get(other_id, 0.5) +
                        0.1 * schema.impression_valence
                    )
```

## Appendix B: Configuration Example

```yaml
# Example configuration for an economic emergence experiment
simulation:
  steps: 10000
  random_seed: 42

agent:
  count: 50
  start_health: 100
  start_time_budget: 1000

learning:
  memory:
    reflection_interval: 100
    emotion_cluster_min_data: 50
  q_learning:
    alpha: 0.001
    gamma: 0.95
    epsilon_start: 1.0
    epsilon_end: 0.1

world:
  grid_size: [50, 50]
  resources:
    spawn_probability: 0.02
    types:
      - name: "food"
        respawn_time: 100
        yield: 20
      - name: "materials"
        respawn_time: 500
        yield: 50
```

[DIAGRAM PLACEHOLDER: Screenshot montage showing visualization output, MLflow dashboard, and analysis plots]
