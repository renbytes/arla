# ARLA Architecture

The ARLA framework is engineered for modularity and extensibility, enabling the construction of sophisticated agent-based simulations. At its core, the architecture leverages the Entity Component System (ECS) pattern to enforce a strict separation between data (Components) and logic (Systems).

This design empowers developers to introduce new agent behaviors, cognitive models, and environmental rules with minimal impact on the core simulation engine, fostering rapid experimentation and research.

## Core Principles

**Data-Oriented Design**: At the heart of ARLA is the ECS pattern. Entities are unique identifiers, Components are pure data structures, and Systems encapsulate all operational logic. This separation makes the simulation state transparent, easy to debug, and simple to serialize.

**Asynchronous Concurrency**: All systems are designed to be non-blocking and are executed concurrently by an asynchronous scheduler. This is critical for performance, especially when integrating I/O-bound operations like calls to Large Language Models (LLMs).

**Decoupled Communication**: Systems operate in isolation and communicate indirectly through a central Event Bus. This event-driven approach ensures that new functionality can be added without modifying existing systems, promoting a highly modular and testable codebase.

**World-Agnostic Engine**: The agent-engine is fundamentally agnostic to the rules of any specific simulation. World-specific logic—such as physics, valid actions, or environmental interactions—is injected into the engine through a set of clearly defined interfaces, ensuring the core is reusable and generic.

## High-Level Diagram

The diagram below illustrates the flow of data and control between the primary layers of the ARLA framework.

```mermaid
graph TD
    subgraph Simulation Layer (e.g., soul_sim)
        A[World-Specific Systems & Components] --> B(Scenario Loader);
        C[Action Implementations] --> B;
    end

    subgraph Core Engine (agent-engine)
        D(Simulation Manager) --> E{Main Loop};
        E --> F[System Manager];
        F --> G((Event Bus));
        F --> H[Async System Runner];
        H --> I[Systems: Reflection, Q-Learning, etc.];
        I -- Publishes Events --> G;
        I -- Reads/Writes Data --> J(SimulationState);
    end

    subgraph Core Abstractions (agent-core)
        J -- Contains --> K[Entities & Components];
        G -- Subscribes to --> L[System Interfaces];
    end

    B -- Populates --> J;

    style D fill:#cde4f9,stroke:#333,stroke-width:2px
    style J fill:#d5f5e3,stroke:#333,stroke-width:2px
```

As shown, a concrete Simulation Layer provides the specific implementations (actions, components) that the generic Core Engine orchestrates. The Core Abstractions provide the foundational interfaces and data structures that ensure loose coupling between all parts of the system.