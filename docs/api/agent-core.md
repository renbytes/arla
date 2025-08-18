# API Reference: agent-core

This section provides the auto-generated API documentation for the `agent-core` library. These are the foundational, world-agnostic classes and interfaces that form the basis of all ARLA simulations.

## Core ECS Classes

These are the central classes of the Entity Component System pattern.

::: agent_core.core.ecs.abstractions.AbstractSimulationState
    options:
      show_root_heading: true
      show_source: false

::: agent_core.core.ecs.component.Component
    options:
      show_root_heading: true
      show_source: false

## Action Interfaces

This is the primary interface that all agent actions must implement.

::: agent_core.agents.actions.action_interface.ActionInterface
    options:
      show_root_heading: true
      show_source: false

## Core Systems Interface

This is the base class that all systems in the `agent-engine` inherit from.

::: agent_engine.simulation.system.System
    options:
      show_root_heading: true
      show_source: false
