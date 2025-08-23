# FILE: simulations/berry_sim/renderer.py

import numpy as np
import imageio
from pathlib import Path
from typing import Any

from .components import (
    PositionComponent,
    HealthComponent,
    WaterComponent,
    RockComponent,
)


class BerryRenderer:
    """Renders the state of the Berry Toxicity simulation grid to an image."""

    def __init__(self, width: int, height: int, output_dir: str, pixel_scale: int = 1):
        self.width = width
        self.height = height
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pixel_scale = pixel_scale

        # Define a color map for all entities
        self.colors = {
            "empty": [25, 25, 25],  # Dark gray for empty space
            "agent": [236, 240, 241],  # White for agents
            "low_health": [192, 57, 43],  # Red for low-health agents
            "red_berry": [231, 76, 60],  # Red
            "blue_berry": [52, 152, 219],  # Blue
            "yellow_berry": [241, 196, 15],  # Yellow
            "water": [41, 128, 185],  # Darker blue for water
            "rock": [127, 140, 141],  # Gray for rocks
        }

    def render_frame(self, simulation_state: Any, tick: int) -> None:
        """Creates and saves a single frame of the simulation."""
        scaled_height = self.height * self.pixel_scale
        scaled_width = self.width * self.pixel_scale
        grid = np.full(
            (scaled_height, scaled_width, 3), self.colors["empty"], dtype=np.uint8
        )

        # Draw terrain first (water and rocks)
        self._draw_terrain(grid, simulation_state, WaterComponent, "water")
        self._draw_terrain(grid, simulation_state, RockComponent, "rock")

        # Draw berries
        self._draw_berries(grid, simulation_state)

        # Draw agents on top
        self._draw_agents(grid, simulation_state)

        frame_path = self.output_path / f"frame_{tick:04d}.png"
        imageio.imwrite(frame_path, grid)

    def _draw_pixel_block(self, grid, x, y, color):
        """Draws a scaled block of pixels on the grid."""
        y_start = y * self.pixel_scale
        y_end = y_start + self.pixel_scale
        x_start = x * self.pixel_scale
        x_end = x_start + self.pixel_scale
        grid[y_start:y_end, x_start:x_end] = color

    def _draw_terrain(self, grid, sim_state, component_type, color_key):
        """Helper to draw static terrain elements like water or rocks."""
        # This assumes terrain entities have a PositionComponent, which they don't.
        # A better way is to get the positions from the environment.
        env = sim_state.environment
        locations = set()
        if color_key == "water" and hasattr(env, "water_locations"):
            locations = env.water_locations
        elif color_key == "rock" and hasattr(env, "rock_locations"):
            locations = env.rock_locations

        for pos in locations:
            self._draw_pixel_block(grid, pos[0], pos[1], self.colors[color_key])

    def _draw_berries(self, grid, sim_state):
        """Helper to draw berries from the environment."""
        env = sim_state.environment
        if hasattr(env, "berry_locations"):
            for pos, berry_type in env.berry_locations.items():
                color = self.colors.get(f"{berry_type}_berry", self.colors["empty"])
                self._draw_pixel_block(grid, pos[0], pos[1], color)

    def _draw_agents(self, grid, sim_state):
        """Helper to draw agents, coloring them by health."""
        entities = sim_state.get_entities_with_components(
            [PositionComponent, HealthComponent]
        )
        for _, components in entities.items():
            pos_comp = components.get(PositionComponent)
            health_comp = components.get(HealthComponent)

            if pos_comp and health_comp:
                health_ratio = health_comp.current_health / health_comp.initial_health
                # Agents turn red if their health is below 30%
                color = (
                    self.colors["agent"]
                    if health_ratio > 0.3
                    else self.colors["low_health"]
                )
                self._draw_pixel_block(grid, pos_comp.x, pos_comp.y, color)
