import random
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Type

# Import our new, high-performance runner from the agent_concurrent package.
from agent_concurrent import ParallelSystemRunner


# 1. DEFINE COMPONENTS (The "Data")
# ---------------------------------
@dataclass
class Position:
    x: float = 0.0
    y: float = 0.0


@dataclass
class Velocity:
    dx: float = 0.0
    dy: float = 0.0


# 2. DEFINE THE SIMULATION STATE
# ---------------------------------
class SimulationState:
    """A simple ECS-style state manager."""

    def __init__(self):
        self.entities: Dict[int, Dict[Type, object]] = {}
        self.next_entity_id = 0

    def add_entity(self, components: List[object]) -> int:
        entity_id = self.next_entity_id
        self.entities[entity_id] = {type(c): c for c in components}
        self.next_entity_id += 1
        return entity_id

    def get_entities_with_components(self, *component_types: Type) -> Set[int]:
        """Finds all entity IDs that have all the specified component types."""
        required_types = set(component_types)
        return {
            entity_id
            for entity_id, components in self.entities.items()
            if required_types.issubset(components.keys())
        }


# 3. DEFINE SYSTEMS (The "Logic")
# ---------------------------------
class MovementSystem:
    """Updates entity positions based on their velocity."""

    def update(self, state: SimulationState):
        # Simulate a heavy, CPU-bound task
        time.sleep(0.1)

        entity_ids = state.get_entities_with_components(Position, Velocity)
        for entity_id in entity_ids:
            pos = state.entities[entity_id][Position]
            vel = state.entities[entity_id][Velocity]
            pos.x += vel.dx
            pos.y += vel.dy


class CollisionSystem:
    """A second system to demonstrate parallelism."""

    def update(self, state: SimulationState):
        # Simulate another heavy, CPU-bound task
        time.sleep(0.1)

        # In a real system, this would check for collisions.
        # For this demo, we just need it to run in parallel with MovementSystem.
        _ = state.get_entities_with_components(Position)


class PrintSystem:
    """Prints the position of a specific entity."""

    def update(self, state: SimulationState):
        if 0 in state.entities:
            pos = state.entities[0][Position]
            print(f"Tick: Entity 0 is at ({pos.x:.2f}, {pos.y:.2f})")


# 4. MAIN SIMULATION SCRIPT
# ---------------------------------
def main():
    """Sets up and runs the simulation."""
    print("🚀 Starting ARLA simulation with Rust core...")

    # -- Setup --
    state = SimulationState()
    for _ in range(500):
        state.add_entity(
            [
                Position(x=random.uniform(-100, 100), y=random.uniform(-100, 100)),
                Velocity(dx=random.uniform(-1, 1), dy=random.uniform(-1, 1)),
            ]
        )

    systems = [
        MovementSystem(),
        CollisionSystem(),
        PrintSystem(),
    ]

    # Instantiate our Rust-based runner!
    runner = ParallelSystemRunner(systems)

    # -- Run Loop --
    num_ticks = 10
    start_time = time.perf_counter()

    for _ in range(num_ticks):
        runner.run(state)

    end_time = time.perf_counter()

    # -- Report Performance --
    total_time = end_time - start_time
    avg_tick_time = total_time / num_ticks
    ticks_per_sec = num_ticks / total_time

    print("\n✅ Simulation complete.")
    print("-" * 30)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per tick: {avg_tick_time:.3f} seconds")
    print(f"Ticks per second (TPS): {ticks_per_sec:.2f}")
    print("-" * 30)
    print("Note: Each tick includes two 0.1s simulated workloads. A sequential")
    print("runner would take >0.2s per tick. The parallel runner is much faster.")


if __name__ == "__main__":
    main()
