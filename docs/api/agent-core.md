# API Reference: agent-core

The `agent-core` library provides foundational, world-agnostic classes and interfaces that form the backbone of all ARLA simulations. These abstractions ensure consistency and enable modular development across different simulation environments.

!!! info "Package Overview"
    `agent-core` defines the contracts and base implementations that all ARLA simulations must follow. Think of it as the "constitution" for the ARLA ecosystem, establishing the fundamental patterns that enable modularity and extensibility.

## Core ECS Classes

The Entity-Component-System pattern forms the architectural foundation of ARLA. These classes provide the essential building blocks for simulation state management and agent behavior.

### SimulationState

The central hub for all simulation data, managing entities and their components with efficient queries and updates.

::: agent_core.core.ecs.abstractions.AbstractSimulationState
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_style: google
      docstring_section_style: table

**Core Capabilities:**

- **Entity Management**: Create, destroy, and query entities efficiently
- **Component Storage**: Type-safe component attachment and retrieval
- **Batch Operations**: Optimized queries for entities with specific component combinations
- **State Persistence**: Serialization support for checkpointing and analysis

### Component Base Class

Pure data containers that define entity properties and state.

::: agent_core.core.ecs.component.Component
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_style: google
      docstring_section_style: table

**Design Principles:**

- Components should contain **only data**, no business logic
- Implement `to_dict()` for serialization and persistence
- Include `validate()` for data integrity checking
- Use type hints for all attributes

**Example Component:**

```python
class HealthComponent(Component):
    """Manages agent health and damage tracking."""
    
    def __init__(self, max_health: int = 100):
        self.max_health = max_health
        self.current_health = max_health
        self.last_damage_tick = 0
        self.damage_history: List[int] = []
    
    @property
    def health_percentage(self) -> float:
        """Current health as percentage of maximum."""
        return self.current_health / self.max_health
    
    @property
    def is_critical(self) -> bool:
        """True if health is below 25% of maximum."""
        return self.health_percentage < 0.25
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_health": self.max_health,
            "current_health": self.current_health,
            "last_damage_tick": self.last_damage_tick,
            "damage_history": self.damage_history
        }
    
    def validate(self, entity_id: str) -> Tuple[bool, List[str]]:
        errors = []
        if self.current_health < 0:
            errors.append("Health cannot be negative")
        if self.max_health <= 0:
            errors.append("Max health must be positive")
        return len(errors) == 0, errors
```

---

## Action Interfaces

Actions define the behaviors available to agents. The action system enables dynamic behavior generation and supports machine learning integration.

### ActionInterface

The contract that all agent actions must implement, ensuring consistent integration with the simulation engine.

::: agent_core.agents.actions.action_interface.ActionInterface
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_style: google
      docstring_section_style: table

**Action Lifecycle:**

```mermaid
graph LR
    A[Discovery] --> B[Parameter Generation]
    B --> C[Action Selection]
    C --> D[Execution]
    D --> E[System Processing]
    E --> F[Outcome Calculation]
```

1. **Discovery**: `@action_registry.register` makes actions available
2. **Parameter Generation**: `generate_possible_params()` creates action variants
3. **Selection**: Decision systems choose from available actions
4. **Execution**: `execute()` returns outcome, systems handle state changes
5. **Learning**: `get_feature_vector()` enables ML model training

**Implementation Guidelines:**

- Keep `execute()` lightweight - delegate actual work to Systems
- Generate parameters based on current world state
- Return meaningful ActionOutcome messages for debugging
- Design feature vectors for your learning algorithms

---

## Provider Interfaces

Provider interfaces enable world-agnostic systems to access world-specific data through dependency injection, maintaining the separation between engine and simulation logic.

### Core Provider Pattern

The provider pattern bridges the gap between world-agnostic cognitive systems and world-specific data:

```python
class VitalityMetricsProvider(VitalityMetricsProviderInterface):
    """Bridges health/energy systems with cognitive architecture."""
    
    def get_normalized_vitality_metrics(
        self, 
        entity_id: str, 
        components: Dict[Type[Component], Component], 
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        health_comp = components.get(HealthComponent)
        energy_comp = components.get(EnergyComponent)
        
        if not health_comp or not energy_comp:
            return {"health_norm": 0.5, "energy_norm": 0.5, "fatigue_norm": 0.5}
        
        return {
            "health_norm": health_comp.current / health_comp.max_health,
            "energy_norm": energy_comp.current / energy_comp.max_energy,
            "fatigue_norm": 1.0 - (energy_comp.current / energy_comp.max_energy)
        }
```

**Available Provider Interfaces:**

<div class="grid cards" markdown>

-   **VitalityMetricsProvider**

    ---

    Provides normalized health and energy data for cognitive systems.

-   **NarrativeContextProvider**

    ---

    Supplies contextual information for LLM-based reflection and planning.

-   **StateEncoderProvider**

    ---

    Encodes simulation state into feature vectors for machine learning.

-   **MemoryAccessProvider**

    ---

    Manages filtered access to agent memories and experiences.

</div>

---

## System Base Classes

Foundation classes for implementing simulation logic and cognitive systems.

### System Base Class

The parent class for all simulation systems, providing event bus access and lifecycle management.

::: agent_engine.simulation.system.System
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      docstring_style: google
      docstring_section_style: table

**System Development Pattern:**

```python
class ExampleSystem(System):
    """Example system demonstrating best practices."""
    
    def __init__(self, simulation_state, config, cognitive_scaffold):
        super().__init__(simulation_state, config, cognitive_scaffold)
        
        # Subscribe to relevant events
        if self.event_bus:
            self.event_bus.subscribe("target_event", self.handle_event)
            self.event_bus.subscribe("tick_completed", self.on_tick_complete)
    
    def handle_event(self, event_data: Dict[str, Any]) -> None:
        """Process specific events with proper error handling."""
        try:
            entity_id = event_data.get("entity_id")
            if not entity_id:
                return
            
            # Process event logic here
            self._process_event_logic(entity_id, event_data)
            
        except Exception as e:
            print(f"Error handling event in {self.__class__.__name__}: {e}")
    
    async def update(self, current_tick: int) -> None:
        """Main system update loop called each simulation tick."""
        # Periodic processing logic here
        entities = self.simulation_state.get_entities_with_components([
            RequiredComponent
        ])
        
        for entity_id, components in entities.items():
            await self._process_entity(entity_id, components, current_tick)
    
    def _process_entity(self, entity_id: str, components: Dict, tick: int):
        """Process individual entity - separate method for testing."""
        pass
```

---

## Configuration Management

ARLA uses Pydantic for type-safe, validated configuration management across all simulations.

### Configuration Schema Pattern

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class AgentConfig(BaseModel):
    """Configuration for agent behavior and properties."""
    max_health: int = Field(default=100, gt=0, description="Maximum health points")
    learning_rate: float = Field(default=0.01, gt=0, le=1, description="Learning rate for Q-learning")
    memory_capacity: int = Field(default=1000, gt=0, description="Size of agent memory buffer")
    
class EnvironmentConfig(BaseModel):
    """Configuration for world environment settings."""
    grid_size: tuple[int, int] = Field(default=(50, 50), description="World dimensions")
    resource_spawn_rate: float = Field(default=0.02, ge=0, le=1, description="Resource spawn probability per tick")
    
class SimulationConfig(BaseModel):
    """Top-level simulation configuration."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    max_ticks: int = Field(default=1000, gt=0, description="Maximum simulation duration")
    random_seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
```

**Configuration Benefits:**

- **Type Safety**: Catch configuration errors at startup
- **Validation**: Automatic constraint checking with clear error messages
- **Documentation**: Self-documenting configuration schemas
- **IDE Support**: Autocompletion and type checking during development

---

## Migration Guide

### From Dictionary-Based Configuration

**Before:**
```python
# Fragile dictionary access with no validation
health = config.get("agent", {}).get("health", {}).get("max", 100)
learning_rate = config.get("learning", {}).get("rate", 0.01)
```

**After:**
```python
# Type-safe access with IDE support
health = config.agent.max_health
learning_rate = config.agent.learning_rate
```

### Component Design Best Practices

**Avoid Logic in Components:**
```python
class BadComponent(Component):
    def update_health(self, damage):  # Logic belongs in Systems!
        self.health -= damage
        if self.health <= 0:
            self.trigger_death()  # Side effects in components are bad
```

**Prefer Pure Data:**
```python
class GoodComponent(Component):
    def __init__(self, max_health: int = 100):
        self.max_health = max_health
        self.current_health = max_health  # Just data storage
        self.damage_taken = 0
    
    # Properties for computed values are acceptable
    @property
    def is_alive(self) -> bool:
        return self.current_health > 0
```

---

## Performance Considerations

### Component Queries

- Use `get_entities_with_components()` for bulk operations
- Cache component references when processing multiple entities
- Prefer specific component queries over entity iteration

```python
# Efficient bulk processing
entities = simulation_state.get_entities_with_components([
    HealthComponent, PositionComponent
])

for entity_id, components in entities.items():
    health_comp = components[HealthComponent]
    pos_comp = components[PositionComponent]
    # Process with cached references
```

### Memory Management

- Components are lightweight - favor composition over inheritance
- Use `__slots__` for frequently instantiated components
- Clean up references in component destructors

```python
class OptimizedComponent(Component):
    __slots__ = ['x', 'y', 'timestamp']  # Reduces memory overhead
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.timestamp = time.time()
```

### Event Bus Usage

- Subscribe to specific events, not broad categories
- Unsubscribe systems that are no longer needed
- Use event data efficiently to avoid serialization overhead

```python
# Efficient event handling
def __init__(self, simulation_state, config, cognitive_scaffold):
    super().__init__(simulation_state, config, cognitive_scaffold)
    
    # Subscribe to specific, relevant events only
    if self.event_bus:
        self.event_bus.subscribe("agent_death", self.handle_death)
        self.event_bus.subscribe("resource_depleted", self.handle_depletion)
        # Avoid subscribing to "all_events" or overly broad categories
```