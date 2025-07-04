# Agent Core

**A foundational, world-agnostic library for building modular and extensible agent-based simulations.**

Agent Core provides the essential building blocks for creating complex AI agent simulations. It is designed to be lightweight and unopinionated about the specific rules or environment of a simulation, focusing instead on providing a stable, decoupled architecture through a robust Entity-Component-System (ECS) pattern and a set of clear interfaces.

---

### Key Features

- **Decoupled Architecture:** Core logic is separated from simulation-specific implementations using a system of abstract interfaces.
- **Entity-Component-System (ECS):** A highly performant and flexible pattern for managing agent and world state.
- **Event-Driven:** Uses a simple event bus for communication between systems, reducing tight coupling.
- **Extensible Action System:** A central action registry allows for the dynamic addition of new agent capabilities without modifying core engine code.
- **Agnostic by Design:** Contains no world-specific logic (e.g., grid movement, combat rules), making it adaptable to any type of simulation.

---

### Core Concepts

- **`Component`:** A pure data container (e.g., `TimeBudgetComponent`, `IdentityComponent`). Components hold the state of an entity but contain no logic.
- **`System`:** A class that operates on entities possessing a specific set of components. All simulation logic lives within systems (e.g., `AffectSystem`, `GoalSystem`). These are implemented in the higher-level `agent-engine`.
- **`Interface`:** An abstract base class that defines a contract for simulation-specific logic. For example, `StateEncoderInterface` defines *how* an agent's state should be converted into a feature vector, but the concrete implementation lives in the final simulation application.
- **`ActionRegistry`:** A global, singleton object where all possible agent actions (e.g., `Move`, `Extract`) are registered. This allows any system to query and generate valid actions for an agent.

---

### How It's Used

This library is not meant to be run standalone. It serves as a dependency for a higher-level simulation engine, such as `agent-engine`. The engine implements the interfaces provided by `agent-core` and builds the specific simulation logic.

**Example Workflow:**

1.  An `agent-engine` project lists `agent-core` as a dependency.
2.  It creates concrete implementations of interfaces like `StateEncoderInterface` and `RewardCalculatorInterface`.
3.  It defines world-specific `Action` classes (e.g., `MoveAction`, `CommunicateAction`) and registers them with the `action_registry`.
4.  It builds `System` classes (e.g., `MovementSystem`) that use the core components and the implemented interfaces to run the simulation.

---

### Development

To install the necessary dependencies for local development and testing, run:

```bash
# Navigate to the agent-core project root
pip install -e .[dev]
```

To run the unit tests:

```bash
pytest
```

To run the static type checker:

```bash
mypy src
```
