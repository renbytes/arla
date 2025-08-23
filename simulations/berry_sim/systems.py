# simulations/berry_sim/systems.py

import random
from typing import Any, Dict, List, Type

from agent_core.agents.actions.base_action import ActionOutcome
from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

from .components import HealthComponent, PositionComponent
from .environment import BerryWorldEnvironment
from .metrics.causal_metrics_calculator import CausalMetricsCalculator
from .renderer import BerryRenderer


class BerrySpawningSystem(System):
    """Handles the spawning of berries according to the experimental protocol."""

    async def update(self, current_tick: int) -> None:
        env = self.simulation_state.environment
        if not isinstance(env, BerryWorldEnvironment):
            return

        spawn_config = self.config.environment.spawning

        # Phase 1: Learning Period (0-1000 ticks)
        if 0 <= current_tick < 1000:
            self._spawn_berries(env, spawn_config, "phase1")
        # Phase 2: Novel Context Test (1000-1100 ticks)
        elif 1000 <= current_tick < 1100:
            self._spawn_berries(env, spawn_config, "phase2")
        # Phase 3: Validation Period (1100-1600 ticks)
        else:
            self._spawn_berries(env, spawn_config, "phase3")

    def _spawn_berries(self, env: BerryWorldEnvironment, config: Any, phase: str):
        # Red Berries
        if random.random() < config.red_rate:
            pos = env.get_random_empty_cell()
            if pos:
                env.berry_locations[pos] = "red"

        # Blue Berries
        if random.random() < config.blue_rate:
            if phase in ["phase1", "phase3"]:  # Spawn AWAY from water
                for _ in range(100):
                    pos = env.get_random_empty_cell()
                    if pos and not env.is_near_feature(pos, env.water_locations, 3):
                        env.berry_locations[pos] = "blue"
                        break
            else:  # Phase 2: Spawn NEAR water
                for _ in range(100):
                    pos = env.get_random_empty_cell()
                    if pos and env.is_near_feature(pos, env.water_locations, 2):
                        env.berry_locations[pos] = "blue"
                        break

        # Yellow Berries
        if random.random() < config.yellow_rate and env.rock_locations:
            rock_pos = random.choice(list(env.rock_locations))
            for _ in range(10):  # Try to find a spot near the rock
                dx, dy = random.randint(-2, 2), random.randint(-2, 2)
                pos = (rock_pos[0] + dx, rock_pos[1] + dy)
                if env.is_valid_position(pos) and not env.is_occupied(pos):
                    env.berry_locations[pos] = "yellow"
                    break


class ConsumptionSystem(System):
    """Handles the consequences of an agent eating a berry."""

    def __init__(self, sim_state: Any, config: Any, scaffold: Any):
        super().__init__(sim_state, config, scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_eat_berry_action", self.on_eat_berry)

    def on_eat_berry(self, event_data: Dict[str, Any]):
        entity_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        current_tick = event_data["current_tick"]

        health_comp = self.simulation_state.get_component(entity_id, HealthComponent)
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        env = self.simulation_state.environment

        if not all([health_comp, pos_comp, isinstance(env, BerryWorldEnvironment)]):
            self._publish_outcome(
                event_data, success=False, reward=-1.0, message="Missing components."
            )
            return

        berry_type = env.berry_locations.pop(pos_comp.position, None)
        if berry_type != params.get("berry_type"):
            self._publish_outcome(
                event_data, success=False, reward=-0.1, message="Berry disappeared."
            )
            return

        health_effect = env.get_berry_toxicity(
            berry_type, pos_comp.position, current_tick
        )
        health_comp.current_health += health_effect
        health_comp.current_health = min(
            health_comp.current_health, health_comp.initial_health
        )

        self._publish_outcome(
            event_data,
            success=True,
            reward=health_effect,
            message=f"Ate {berry_type} berry.",
        )

    def _publish_outcome(
        self, event_data: Dict[str, Any], success: bool, reward: float, message: str
    ):
        event_data["action_outcome"] = ActionOutcome(
            success, message, base_reward=reward
        )
        event_data["original_action_plan"] = event_data.pop("action_plan_component")
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int):
        pass


class MovementSystem(System):
    """Handles agent movement, including a simple heuristic to move towards berries."""

    def __init__(self, sim_state: Any, config: Any, scaffold: Any):
        super().__init__(sim_state, config, scaffold)
        if self.event_bus:
            self.event_bus.subscribe("execute_move_action", self.on_move)

    def on_move(self, event_data: Dict[str, Any]):
        entity_id = event_data["entity_id"]
        params = event_data["action_plan_component"].params
        pos_comp = self.simulation_state.get_component(entity_id, PositionComponent)
        env = self.simulation_state.environment

        if not all([pos_comp, isinstance(env, BerryWorldEnvironment)]):
            self._publish_outcome(
                event_data, success=False, reward=-1.0, message="Missing components."
            )
            return

        target_pos = params["target_pos"]
        if not env.is_valid_position(target_pos) or env.is_occupied(target_pos):
            self._publish_outcome(
                event_data,
                success=False,
                reward=-0.1,
                message="Target invalid or occupied.",
            )
            return

        old_pos = pos_comp.position
        pos_comp.x, pos_comp.y = target_pos
        env.update_entity_position(entity_id, old_pos, target_pos)
        self._publish_outcome(
            event_data, success=True, reward=0.0, message="Move successful."
        )

    def _publish_outcome(
        self, event_data: Dict[str, Any], success: bool, reward: float, message: str
    ):
        event_data["action_outcome"] = ActionOutcome(
            success, message, base_reward=reward
        )
        event_data["original_action_plan"] = event_data.pop("action_plan_component")
        if self.event_bus:
            self.event_bus.publish("action_outcome_ready", event_data)

    async def update(self, current_tick: int):
        pass


class VitalsSystem(System):
    """Checks agent vitals and handles deactivation (death)."""

    REQUIRED_COMPONENTS: List[Type[Component]] = [HealthComponent, TimeBudgetComponent]

    async def update(self, current_tick: int) -> None:
        agents = self.simulation_state.get_entities_with_components(
            self.REQUIRED_COMPONENTS
        )
        env = self.simulation_state.environment

        for agent_id, components in agents.items():
            health_comp = components.get(HealthComponent)
            time_comp = components.get(TimeBudgetComponent)

            if not time_comp or not health_comp or not time_comp.is_active:
                continue

            if health_comp.current_health <= 0:
                time_comp.is_active = False

                if isinstance(env, BerryWorldEnvironment):
                    env.remove_entity(agent_id)

                if self.event_bus:
                    self.event_bus.publish(
                        "agent_deactivated",
                        {"entity_id": agent_id, "current_tick": current_tick},
                    )


class CausalMetricTrackerSystem(System):
    """Listens to events to update the state of the causal metrics calculator."""

    REQUIRED_COMPONENTS: List[Type[Component]] = []

    def __init__(
        self,
        simulation_state: Any,
        config: Any,
        cognitive_scaffold: Any,
        calculator: CausalMetricsCalculator,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)
        self.calculator = calculator
        if self.event_bus:
            self.event_bus.subscribe("action_executed", self.on_action_executed)

    def on_action_executed(self, event_data: Dict[str, Any]) -> None:
        """Passes event data to the calculator to update its internal state."""
        self.calculator.update_with_event(event_data, self.simulation_state)

    async def update(self, current_tick: int) -> None:
        pass


class RenderingSystem(System):
    """A system that renders the simulation state to an image at each tick."""

    def __init__(
        self,
        simulation_state: Any,
        config: Any,
        cognitive_scaffold: Any,
    ):
        super().__init__(simulation_state, config, cognitive_scaffold)

        # Initialize the renderer with parameters from the config
        env_params = config.environment.get("params", {})
        width = env_params.get("width", 50)
        height = env_params.get("height", 50)

        render_config = config.get("rendering", {})
        output_dir = render_config.get("output_directory", "data/renders/default_berry")
        pixel_scale = render_config.get("pixel_scale", 1)

        self.renderer = BerryRenderer(width, height, output_dir, pixel_scale)
        print(
            f"🎨 RenderingSystem initialized. Frames will be saved to '{output_dir}'."
        )

    async def update(self, current_tick: int) -> None:
        """On each tick, render a new frame."""
        self.renderer.render_frame(self.simulation_state, current_tick)
