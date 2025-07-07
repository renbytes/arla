# Decoupling Controllability in the AffectSystem

This document details the latest architectural refinement within the Affective Reflective Learning Architecture (ARLA), focusing on how the AffectSystem in agent-engine has been further decoupled from world-specific components, particularly concerning the concept of "controllability."

## The Problem: Lingering World-Specific Coupling

Even after abstracting vitality metrics, the AffectSystem still retained a direct dependency on `PositionComponent` and `FailedStatesComponent`. This was evident in how it calculated "controllability":

**Direct Component Access**: The AffectSystem directly retrieved `FailedStatesComponent` and `PositionComponent`.

**World-Specific Logic**: It then used `pos_comp.position` as a key in `failed_states_comp.tracker` to determine `failed_count`. This implied that "failed states" were tracked per location, a concept specific to a physical, spatial world simulation.

**Violation of World-Agnosticism**: A core cognitive system like AffectSystem (responsible for emotional appraisal) should not need to know about spatial positions or location-specific failure counts. Its input for "controllability" should be a generalized metric, not derived from world-specific mechanics.

This coupling limited the AffectSystem's reusability in non-spatial or abstract environments.

## The Solution: The ControllabilityProviderInterface

To achieve complete decoupling for the AffectSystem, we introduced another abstract interface: `ControllabilityProviderInterface`.

### Where it Lives

`src/agent_core/environment/controllability_provider_interface.py`: This new file in the agent-core library defines the contract for providing a world-agnostic controllability score.

```python
# src/agent_core/environment/controllability_provider_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.core.ecs.component import Component
    from agent_engine.simulation.simulation_state import SimulationState

class ControllabilityProviderInterface(ABC):
    @abstractmethod
    def get_controllability_score(
        self,
        entity_id: str,
        components: Dict[Type["Component"], "Component"],
        simulation_state: "SimulationState",
        current_tick: int,
        config: Dict[str, Any]
    ) -> float:
        """
        Calculates a normalized controllability score (0.0 to 1.0) for the agent
        based on world-specific factors.
        """
        raise NotImplementedError
```

### How it Works

```mermaid
graph TD
    subgraph Agent-Soul-Sim (Application Layer)
        A[MyGridWorldEnvironment] --> D
        B[MyGridWorldActionGenerator] --> D
        C[MyGridWorldDecisionSelector] --> D
        E[GridWorldStateNodeEncoder] --> SM
        F[GridWorldVitalityMetricsProvider] --> AS
        G[GridWorldNarrativeContextProvider] --> RS
        I[GridWorldControllabilityProvider] --> AS
        H[World-Specific Systems] --> SM
        D(main.py - Initializes & Registers) --> SM[SimulationManager<br>(agent-engine)]
    end

    subgraph agent-engine (The Brain - World-Agnostic)
        SM --> AS[AffectSystem]
        SM --> CG[CausalGraphSystem]
        SM --> RS[ReflectionSystem]
        SM --> QL[QLearningSystem]
        SM --> GS[GoalSystem]
        SM --> IS[IdentitySystem]
        AS -.-> VMP(VitalityMetricsProviderInterface)
        AS -.-> CP(ControllabilityProviderInterface)
        CG -.-> SNE(StateNodeEncoderInterface)
        RS -.-> NCP(NarrativeContextProviderInterface)
        QL -.-> SE(StateEncoderInterface)
    end

    subgraph agent-core (The Soul - Foundational)
        Comp[Components<br>(AffectComponent, EmotionComponent, GoalComponent, etc.)]
        Interfaces[Interfaces<br>(VitalityMetricsProviderInterface, ControllabilityProviderInterface, etc.)]
        Util[Agnostic Utilities]
    end

    Comp --> AS
    Interfaces --> AS
    Util --> AS
    I --> CP
    F --> VMP
    E --> SNE
    G --> NCP
    CP -.-> Comp
    VMP -.-> Comp
    SNE -.-> Comp
    NCP -.-> Comp
    Comp --> I
    Comp --> F
    Comp --> E
    Comp --> G
```

#### AffectSystem (in agent-engine)

- The AffectSystem no longer imports or directly accesses `FailedStatesComponent` or `PositionComponent`.
- Its `__init__` method now accepts an instance of `ControllabilityProviderInterface` via dependency injection.
- When it needs the controllability score (in `_process_affective_response`), it calls the `get_controllability_score` method on the injected `controllability_provider` instance, passing the necessary components and simulation state.

```python
# src/agent_engine/systems/affect_system.py (Key changes)
# ...
from agent_core.core.ecs.component import (
    AffectComponent, EmotionComponent, GoalComponent,
    # REMOVED: FailedStatesComponent, PositionComponent
)
from agent_core.environment.controllability_provider_interface import ControllabilityProviderInterface # NEW IMPORT

class AffectSystem(System):
    REQUIRED_COMPONENTS: List[Type[CognitiveComponent]] = [AffectComponent, EmotionComponent, GoalComponent]
    # REMOVED: FailedStatesComponent, PositionComponent

    def __init__(
        self,
        simulation_state: SimulationState,
        config: Dict[str, Any],
        cognitive_scaffold: Any,
        vitality_metrics_provider: VitalityMetricsProviderInterface,
        controllability_provider: ControllabilityProviderInterface # NEW INJECTED DEPENDENCY
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.controllability_provider = controllability_provider # Store the injected provider
        # ...

    def _process_affective_response(self, ..., current_tick: int) -> None:
        # ...
        # Get controllability from the injected provider
        controllability = self.controllability_provider.get_controllability_score(
            entity_id=entity_id,
            components=components, # Pass all components for the provider to extract what it needs
            simulation_state=self.simulation_state,
            current_tick=current_tick,
            config=self.config
        )
        # ... emotional dynamics update now uses this abstract 'controllability' ...
```

#### Application Layer (agent-soul-sim)

- A concrete implementation of `ControllabilityProviderInterface` is now created within the agent-soul-sim project (e.g., `src/my_simulation/grid_world_controllability_provider.py`).
- This concrete class contains the actual world-specific logic for calculating the controllability score, accessing components like `FailedStatesComponent` and `PositionComponent` as needed.

```python
# src/my_simulation/grid_world_controllability_provider.py (Example)
# ...
from agent_core.environment.controllability_provider_interface import ControllabilityProviderInterface
from agent_core.core.ecs.component import Component, FailedStatesComponent, PositionComponent
from agent_engine.simulation.simulation_state import SimulationState

class GridWorldControllabilityProvider(ControllabilityProviderInterface):
    def get_controllability_score(
        self,
        entity_id: str,
        components: Dict[Type[Component], Component],
        simulation_state: "SimulationState",
        current_tick: int,
        config: Dict[str, Any]
    ) -> float:
        # This is where world-specific logic lives!
        failed_states_comp = cast(FailedStatesComponent, components.get(FailedStatesComponent))
        pos_comp = cast(PositionComponent, components.get(PositionComponent))

        if not failed_states_comp or not pos_comp:
            return 0.5 # Default if world-specific components are missing

        failed_count = failed_states_comp.tracker.get(pos_comp.position, 0)
        controllability = max(0.1, 1.0 - failed_count * 0.1)
        return float(controllability)
```

#### Simulation Initialization (main.py in agent-soul-sim)

- The `main.py` script is now responsible for instantiating `GridWorldControllabilityProvider`.
- This instance is then passed to the `SimulationManager` when registering the `AffectSystem`.

```python
# src/agent_soul_sim/main.py (Key change)
# ...
from agent_engine.systems.affect_system import AffectSystem
from my_simulation.grid_world_controllability_provider import GridWorldControllabilityProvider # NEW IMPORT

# ...
grid_world_controllability_provider = GridWorldControllabilityProvider()

sim_manager = SimulationManager(...)

sim_manager.register_system(
    AffectSystem,
    vitality_metrics_provider=grid_world_vitality_metrics_provider,
    controllability_provider=grid_world_controllability_provider # INJECTION!
)
# ...
```

## Benefits of This Decoupling

**Complete World-Agnosticism for AffectSystem**: The AffectSystem now operates solely on abstract, generalized metrics (prediction error, social context, vitality, and controllability). It has no direct knowledge of physical locations, health points, or resource counts.

**Enhanced Reusability**: The AffectSystem can be seamlessly integrated into any simulation world, regardless of its underlying mechanics for tracking success, failure, or environmental difficulty.

**Clearer Separation of Concerns**: The responsibility for translating world-specific events into a generalized "controllability" score is correctly placed in the application layer, where the world's rules are defined.

**Improved Maintainability**: Changes to how environmental difficulty or agent failures are tracked in the world simulation only require updates to the application's ControllabilityProvider, leaving the core agent-engine untouched.
