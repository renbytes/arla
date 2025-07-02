ARLA Core
This library provides the core components, interfaces, and data structures for the Affective Reflective Learning Architecture (ARLA). It is a world-agnostic foundation for building cognitive agents.

Overview
arla-core contains the fundamental building blocks for creating intelligent agents:

ECS Abstractions: Defines the base CognitiveComponent and CognitiveSystem classes.

Core Components: Includes world-agnostic data components for an agent's internal state, such as MemoryComponent, EmotionComponent, IdentityComponent, and GoalComponent.

Interfaces: Provides abstract contracts like ActionInterface and EnvironmentInterface to decouple the agent's mind from the world it inhabits.

Utilities: Contains generic tools like the EventBus for decoupled communication and the CognitiveScaffold for LLM interactions.

This library is intended to be used as a dependency by higher-level packages like arla-engine and specific simulation applications.

