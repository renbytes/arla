# Agent Engine

**A simulation engine for running complex, psychologically-grounded AI agents built on the `agent-core` library.**

Agent Engine provides the concrete systems and logic required to run sophisticated agent-based simulations. It leverages the interfaces and components from `agent-core` to implement cognitive processes like emotional appraisal, identity formation, and goal management. This engine is designed to be the "brains" of a simulation, orchestrating the behavior of agents tick by tick.

---

### Key Features

- **Cognitive Systems:** Includes concrete implementations for key psychological processes:
    - `AffectSystem`: Manages agent emotion and affect using appraisal theory.
    - `IdentitySystem`: Models a multi-domain identity that evolves based on experience.
    - `GoalSystem`: Handles emergent goal creation, selection, and refinement.
    - `CausalGraphSystem`: Allows agents to build a symbolic understanding of cause and effect.
- **Learning Agents:** Implements a `QLearningSystem` that enables agents to learn and improve their decision-making over time.
- **Psychologically-Grounded Models:** Incorporates concepts from psychological literature, such as Lazarus & Folkman's appraisal theory and multi-domain identity models.
- **Decoupled from World-Specifics:** While it contains logic, the engine still relies on interfaces from `agent-core` (like `StateEncoderInterface` and `VitalityMetricsProvider`) to get information about the world, allowing it to remain adaptable to different environments.

---

### Core Concepts

- **`SimulationManager`:** The main orchestrator class that initializes all systems, loads a scenario, and runs the main simulation loop.
- **`System` Implementations:** This engine is primarily composed of concrete `System` classes (e.g., `IdentitySystem`, `GoalSystem`) that operate on the components defined in `agent-core`.
- **Dependency Injection:** The `SimulationManager` and its systems are designed to have their world-specific dependencies (like reward calculators and state encoders) "injected" during setup. This keeps the engine itself reusable.
- **Configuration-Driven:** The engine is designed to be heavily configured via `OmegaConf` files, allowing for easy adjustment of simulation parameters, learning rates, and agent behaviors.

---

### How It's Used

This engine is the runnable part of the simulation architecture. It is typically configured and launched by a final application layer, which provides the last pieces of world-specific logic.

**Example Workflow:**

1.  A final simulation project (e.g., `agent-soul-sim`) lists `agent-engine` as a dependency.
2.  The project defines concrete classes for the interfaces in `agent-core` (e.g., a `MyRewardCalculator` and a `MyStateEncoder`).
3.  It creates a scenario file (e.g., `scenario.json`) that defines the initial agents, their components, and the environment layout.
4.  A main script in the project instantiates the `SimulationManager`, injecting the concrete reward calculator, state encoder, environment, and scenario loader.
5.  The script calls `simulation_manager.run()` to start the simulation.

---

### Development

To install the necessary dependencies for local development and testing, run:

```bash
# Navigate to the agent-engine project root
pip install -e ".[dev]"
```

To run the unit tests:

```bash
pytest
```

To run the static type checker:

```bash
mypy src
```
