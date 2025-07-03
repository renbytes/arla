# ARLA: Affective Reflective Learning Architecture

ARLA is a modular, extensible, and psychologically-grounded platform for building intelligent agents in simulated environments. It's designed to separate core cognitive processes from world-specific simulation logic, allowing for highly reusable agent "brains" across diverse applications.

## Architecture Overview

The ARLA architecture is structured into three main layers, designed for clear separation of concerns and maximum reusability:

- **agent_core**: The Agent's "Soul" (Foundational Data & Interfaces)
- **agent-engine**: The Agent's "Brain" (World-Agnostic Cognitive Logic)
- **agent-soul-sim** (Application Layer): The "Body" & "World" (World-Specific Simulation & Behaviors)

This layered approach ensures that the core agent intelligence (agent-engine) remains portable and reusable, while the specific environment it inhabits (agent-soul-sim) can be swapped out without modifying the fundamental cognitive machinery.

```mermaid
graph TD
    subgraph Agent-Soul-Sim (Application Layer)
        A[MyGridWorldEnvironment] --> D
        B[MyGridWorldActionGenerator] --> D
        C[MyGridWorldDecisionSelector] --> D
        E[GridWorldStateNodeEncoder] --> SM
        F[GridWorldVitalityMetricsProvider] --> SM
        G[GridWorldNarrativeContextProvider] --> SM
        H[World-Specific Systems<br>(e.g., CombatSystem, MovementSystem)] --> SM
        D(main.py - Initializes & Registers) --> SM[SimulationManager<br>(agent-engine)]
    end

    subgraph agent-engine (The Brain - World-Agnostic)
        SM --> AS[AffectSystem]
        SM --> CG[CausalGraphSystem]
        SM --> RS[ReflectionSystem]
        SM --> QL[QLearningSystem]
        SM --> GS[GoalSystem]
        SM --> IS[IdentitySystem]
        SM -.-> Other Engine Systems
        AS -.-> VMP(VitalityMetricsProviderInterface)
        CG -.-> SNE(StateNodeEncoderInterface)
        RS -.-> NCP(NarrativeContextProviderInterface)
        QL -.-> SE(StateEncoderInterface)
        OtherEngineSystems -.-> AllCoreComponents
    end

    subgraph agent_core (The Soul - Foundational)
        Comp[Components<br>(Identity, Memory, Goal, Emotion, Affect, etc.)]
        Interfaces[Interfaces<br>(ActionInterface, EnvironmentInterface,<br>StateNodeEncoderInterface, VitalityMetricsProviderInterface, NarrativeContextProviderInterface)]
        Util[Agnostic Utilities<br>(EventBus, CognitiveScaffold)]
    end

    Comp --> SM
    Comp --> AS
    Comp --> CG
    Comp --> RS
    Comp --> QL
    Comp --> GS
    Comp --> IS
    Interfaces --> AS
    Interfaces --> CG
    Interfaces --> RS
    Interfaces --> QL
    Interfaces --> SE
    Util --> SM
    Util --> AS
    Util --> CG
    Util --> RS
    Util --> QL
    Util --> GS
    Util --> IS

    VMP -.-> Comp
    SNE -.-> Comp
    NCP -.-> Comp
    SE -.-> Comp
```

## 1. agent_core: The Agent's "Soul"

This library is the foundational layer. It is designed to be lightweight, abstract, and completely world-agnostic. It defines the universal "nouns" and "contracts" that any ARLA agent operates with.

### Key Characteristics:

**Defines the "Nouns" (Components)**: Contains core data structures (Component subclasses) representing an agent's internal, psychological state. These concepts are true for any agent, regardless of its specific world.

*Examples*: `IdentityComponent`, `MemoryComponent`, `GoalComponent`, `EmotionComponent`, `AffectComponent`, `ActionPlanComponent`.

**Establishes Contracts (Interfaces)**: Defines abstract interfaces that act as contracts for communication between layers. These ensure decoupling.

*Core Interfaces*: `ActionInterface`, `EnvironmentInterface`.

*New Decoupling Interfaces*:
- `StateNodeEncoderInterface`: For transforming world-specific states into generalized causal graph nodes.
- `VitalityMetricsProviderInterface`: For providing normalized, abstract well-being metrics for affective appraisal.
- `NarrativeContextProviderInterface`: For generating a textual narrative context from mixed cognitive and world-specific states for LLM reflection.

**Provides Agnostic Utilities**: Includes essential, reusable tools not tied to specific logic.

*Examples*: `EventBus` for decoupled communication, `CognitiveScaffold` for interfacing with Large Language Models.

In essence, agent_core is your portable definition of an agent's fundamental self and the universal ways it can interact with a generic "world" and its own internal state.

## 2. agent-engine: The Agent's "Brain"

This library sits on top of agent_core and contains the "verbs" of your architecture. It implements the logic for how an agent thinks, learns, and feels, without any knowledge of the specific world it inhabits.

### Key Characteristics:

**Contains World-Agnostic Logic (Systems)**: Houses all the System classes that process the Components from agent_core.

*Examples*: `AffectSystem` (manages emotions), `ReflectionSystem` (handles metacognition), `QLearningSystem` (manages decision-model updates), `IdentitySystem` (evolves the agent's self-concept), `GoalSystem`, `CausalGraphSystem`.

**Orchestrates the Simulation**: Includes the `SimulationManager`, which is the master clock responsible for running the main update loop, and the `SystemManager`, which orchestrates the execution of all registered systems.

**Decoupled via Dependency Injection**: This is the most powerful feature. The engine defines abstract interfaces (like `StateEncoder`, `RewardCalculator`, and now the new `StateNodeEncoderInterface`, `VitalityMetricsProviderInterface`, `NarrativeContextProviderInterface`) that must be "plugged in" by the final application. This makes the engine incredibly flexible and reusable. The agent-engine systems themselves are blind to "health" or "grid positions"; they only see the abstract data provided by the injected interfaces.

In summary, agent-engine is your reusable cognitive processing unit. It provides the sophisticated machinery for running any ARLA-based agent, but it relies on the final application (agent-soul-sim) to provide the "drivers" for the specific world it's operating in.

## 3. agent-soul-sim: The Agent's "Body" & "World" (Application Layer)

This is your specific simulation application. It consumes the agent_core and agent-engine libraries and brings them to life within a defined world. This layer is responsible for implementing the concrete, world-specific logic.

### Key Responsibilities:

**Concrete Implementations of Interfaces**: Provides concrete classes for all interfaces defined in agent_core and required by agent-engine.

*Examples*: `GridWorldEnvironment`, `GridWorldStateEncoder`, `GridWorldRewardCalculator`, `GridWorldStateNodeEncoder`, `GridWorldVitalityMetricsProvider`, `GridWorldNarrativeContextProvider`.

**World-Specific Components & Systems**: Defines components (e.g., `HealthComponent`, `PositionComponent`, `InventoryComponent`) and systems (e.g., `CombatSystem`, `MovementSystem`, `ResourceGatheringSystem`) that are unique to the simulation world.

**Simulation Initialization**: Sets up the `SimulationManager`, injects all necessary concrete implementations, and registers all cognitive and world-specific systems.

## How it All Comes Together (The Workflow)

The enabled workflow for developing with ARLA is now that of a professional, extensible software platform:

### 1. Install the core libraries
A developer `pip install`s both `agent_core` and `agent-engine`. These are pre-built, world-agnostic packages.

### 2. Develop the world
In their own application repository (e.g., your `agent-soul-sim` project), they create:

- **Concrete implementations** of the `EnvironmentInterface`, `StateEncoder`, `RewardCalculator`, and crucially, the new `StateNodeEncoderInterface`, `VitalityMetricsProviderInterface`, and `NarrativeContextProviderInterface`. These bridge the abstract agent-engine to the concrete world.

- **Any world-specific Components** (e.g., `HealthComponent`, `PositionComponent`, `ResourceComponent`).

- **Any world-specific Systems** (e.g., `CombatSystem`, `MovementSystem`, `InteractionSystem`) that operate on these world-specific components and/or interact with cognitive components.

### 3. Initialize and Run
They write a `main.py` script that:

- **Initializes** the `SimulationManager` from agent-engine.

- **Injects** their concrete environment, state encoder, reward calculator, and the new world-specific data providers (`StateNodeEncoder`, `VitalityMetricsProvider`, `NarrativeContextProvider`) into the manager and relevant engine systems during system registration.

- **Registers** the world-agnostic systems from agent-engine (e.g., `AffectSystem`, `ReflectionSystem`, `CausalGraphSystem`, `GoalSystem`, `IdentitySystem`, `QLearningSystem`).

- **Registers** their own world-specific systems.

- **Runs** the simulation. The engine takes over, orchestrating all the complex cognitive and logical updates in a completely decoupled manner.

This architecture ensures that the intellectual property of your "cognitive brain" (agent-engine) remains separate and reusable, while the specific "game" or "world" (agent-soul-sim) it operates in can be developed independently and swapped out as needed.
