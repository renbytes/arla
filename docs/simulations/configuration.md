# ARLA Project: New Simulation Configuration Guide

## 1. Overview

This guide outlines the standardized configuration system for all simulations within the ARLA project. The framework uses **Pydantic** for configuration management to ensure that all simulations are robust, type-safe, and easier to maintain. By following this pattern, you can catch configuration errors at startup, long before they cause problems deep within a simulation run.

## 2. Creating Your Configuration Schema

For any new simulation (e.g., `your_new_sim`), you must define its complete configuration structure in a single, hierarchical Pydantic model.

- **Action:** Create a `schemas.py` file within your simulation's config directory (e.g., `simulations/your_new_sim/config/schemas.py`).
- **Implementation:** Inside this file, define a main Pydantic `BaseModel` (e.g., `YourNewSimAppConfig`) that contains all possible configuration parameters, nested into logical sub-models (like `AgentConfig`, `EnvironmentConfig`, etc.).

This schema file becomes the **single source of truth** for your simulation's configuration. It serves as explicit, enforceable documentation, defining every parameter, its data type, and any validation rules (e.g., `Field(gt=0)`).

### Example Schema Structure

```python
# simulations/your_new_sim/config/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class AgentConfig(BaseModel):
    """Configuration for agent behavior and properties."""
    max_health: int = Field(default=100, gt=0, description="Maximum health points")
    learning_rate: float = Field(default=0.01, gt=0, le=1, description="Learning rate for Q-learning")
    memory_size: int = Field(default=1000, gt=0, description="Size of agent memory buffer")

class EnvironmentConfig(BaseModel):
    """Configuration for environment settings."""
    grid_size: int = Field(default=20, gt=0, description="Size of the grid world")
    resource_density: float = Field(default=0.1, ge=0, le=1, description="Density of resources")
    max_ticks: int = Field(default=1000, gt=0, description="Maximum simulation ticks")

class YourNewSimAppConfig(BaseModel):
    """Complete configuration schema for YourNewSim simulation."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    experiment_name: str = Field(default="default_experiment", description="Name of the experiment")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
```

## 3. The Validation Workflow

Your simulation's entry point (e.g., `simulations/your_new_sim/run.py`) is responsible for loading and validating the configuration. The standard workflow is:

1. Load a base YAML configuration file (e.g., `simulations/your_new_sim/config/base_config.yml`).
2. Use `OmegaConf` to merge any experiment-specific overrides on top of the base config.
3. Pass the resulting dictionary into your Pydantic model (e.g., `config = YourNewSimAppConfig(**config_dict)`).

This step will immediately parse and validate the entire configuration structure. If any required fields are missing, if a value has the wrong type, or if any validation rule fails, Pydantic will raise a detailed `ValidationError` and stop the application.

### Example Validation Code

```python
# simulations/your_new_sim/run.py
import asyncio
from omegaconf import OmegaConf
from pathlib import Path
from .config.schemas import YourNewSimAppConfig

async def setup_and_run(
    run_id: str,
    task_id: str, 
    experiment_id: str,
    config_overrides: dict,
    checkpoint_path: str = None
):
    """Main entry point for running the simulation."""
    
    # 1. Load base configuration
    base_config_path = Path(__file__).parent / "config" / "base_config.yml"
    base_config = OmegaConf.load(base_config_path)
    
    # 2. Merge with any experiment overrides
    final_config_dict = OmegaConf.merge(base_config, config_overrides)
    
    # 3. Validate using Pydantic schema
    try:
        config = YourNewSimAppConfig(**final_config_dict)
        print("✅ Configuration validation successful")
    except ValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        raise
    
    # 4. Run simulation with validated config
    await run_simulation(config, run_id, task_id, experiment_id, checkpoint_path)
```

## 4. How to Use the Configuration in Code

Once validated, the `config` object—an instance of your Pydantic model, **not a dictionary**—should be passed down through your application, starting with the `SimulationManager`.

To access configuration values, you **must** use direct attribute access. This provides full type safety and IDE autocompletion.

### ❌ Old Way (Incorrect):

```python
time_decay = self.config.get("agent", {}).get("dynamics", {}).get("decay", {}).get("time_budget_per_step")
```

### ✅ New Way (Correct):

```python
time_decay = self.config.agent.dynamics.decay.time_budget_per_step
```

### Example Usage in Classes

```python
# Example: Using config in a system class
class YourNewSimSystem(System):
    def __init__(self, simulation_state: SimulationState, config: YourNewSimAppConfig):
        super().__init__(simulation_state, config)
        
        # Type-safe access with IDE autocompletion
        self.max_health = config.agent.max_health
        self.learning_rate = config.agent.learning_rate
        self.grid_size = config.environment.grid_size
        
        if config.debug_mode:
            print(f"Initialized with grid size: {self.grid_size}")
```

## 5. How to Add a New Configuration Parameter

### Step 1: Update the Schema

Open your simulation's `schemas.py` file and add the new field to the appropriate Pydantic model, defining its type and any constraints.

```python
# Add to existing AgentConfig class
class AgentConfig(BaseModel):
    max_health: int = Field(default=100, gt=0)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    memory_size: int = Field(default=1000, gt=0)
    
    # NEW: Add your new parameter
    aggression_level: float = Field(
        default=0.5, 
        ge=0, 
        le=1, 
        description="Agent aggression level in combat"
    )
```

### Step 2: Update the Base Config

Add the new parameter and a default value to your simulation's `base_config.yml` file, ensuring its location matches the structure in your schema.

```yaml
# simulations/your_new_sim/config/base_config.yml
agent:
  max_health: 100
  learning_rate: 0.01
  memory_size: 1000
  aggression_level: 0.5  # NEW: Add here

environment:
  grid_size: 20
  resource_density: 0.1
  max_ticks: 1000

experiment_name: "base_experiment"
debug_mode: false
```

### Step 3: Use in Code

Access your new parameter in any class that receives the `config` object using direct attribute access.

```python
# Example usage in a combat system
class CombatSystem(System):
    def __init__(self, simulation_state: SimulationState, config: YourNewSimAppConfig):
        super().__init__(simulation_state, config)
        self.aggression_level = config.agent.aggression_level
    
    def calculate_damage(self, base_damage: int) -> int:
        # Use the new configuration parameter
        return int(base_damage * (1 + self.aggression_level))
```

## 6. Configuration Best Practices

### ✅ Do's

- **Use descriptive field names** that clearly indicate purpose
- **Add docstrings** to your Pydantic models for documentation
- **Set appropriate validation constraints** (e.g., `gt=0` for positive values)
- **Provide sensible default values** for all optional fields
- **Group related parameters** into logical sub-models
- **Use type hints** for all fields

### ❌ Don'ts

- **Don't use dictionary access** (`.get()`) on validated config objects
- **Don't bypass validation** by directly creating dictionaries
- **Don't mix validated and unvalidated** configuration objects
- **Don't hardcode values** that should be configurable
- **Don't create deeply nested** structures (limit to 3-4 levels)

### Example of Good Configuration Structure

```python
class DynamicsConfig(BaseModel):
    """Agent dynamics and learning parameters."""
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    discount_factor: float = Field(default=0.95, gt=0, le=1)
    exploration_rate: float = Field(default=0.1, ge=0, le=1)

class PhysicalConfig(BaseModel):
    """Agent physical properties."""
    max_health: int = Field(default=100, gt=0)
    movement_speed: float = Field(default=1.0, gt=0)
    vision_range: int = Field(default=5, gt=0)

class AgentConfig(BaseModel):
    """Complete agent configuration."""
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    physical: PhysicalConfig = Field(default_factory=PhysicalConfig)
    name_prefix: str = Field(default="Agent", min_length=1)
```

## 7. Troubleshooting Common Issues

### Validation Errors

When you see validation errors, they typically fall into these categories:

1. **Missing required fields**: Add the field to your YAML config
2. **Type mismatches**: Ensure YAML values match schema types (int vs float vs str)
3. **Constraint violations**: Check Field constraints like `gt=0`, `le=1`
4. **Nested structure issues**: Verify YAML nesting matches Pydantic model structure

### Example Error and Fix

```bash
# Error message
ValidationError: 1 validation error for YourNewSimAppConfig
agent.learning_rate
  ensure this value is greater than 0 (type=value_error.number.not_gt; limit_value=0)
```

```yaml
# Fix: Update base_config.yml
agent:
  learning_rate: 0.01  # Was 0 or negative, now positive
```

This configuration system ensures that your simulations are robust, well-documented, and maintainable while providing excellent developer experience through type safety and IDE support.