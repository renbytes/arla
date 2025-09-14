"""
Defines the renderer for the Tragedy of the Commons simulation.

This class is responsible for creating a visual representation of the
simulation state at each tick and saving it as an image frame.
"""

import numpy as np
import imageio
from pathlib import Path
from typing import Any, cast

from .components import PositionComponent, ResourceComponent
from .environment import CommonsEnvironment


class CommonsRenderer:
    """Renders the state of the Commons simulation grid to an image."""

    def __init__(self, width: int, height: int, output_dir: str, pixel_scale: int = 1):
        self.width = width
        self.height = height
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pixel_scale = pixel_scale

        # Define colors
        self.agent_color = np.array([236, 240, 241], dtype=np.uint8)  # White
        self.grass_min_color = np.array([46, 204, 113], dtype=np.uint8)  # Dark Green
        self.grass_max_color = np.array([22, 160, 133], dtype=np.uint8)  # Darker Green
        self.depleted_color = np.array([210, 179, 134], dtype=np.uint8)  # Brown/Dirt

    def render_frame(self, simulation_state: Any, tick: int) -> None:
        """Creates and saves a single frame of the simulation."""
        scaled_height = self.height * self.pixel_scale
        scaled_width = self.width * self.pixel_scale

        grid = np.full(
            (scaled_height, scaled_width, 3), self.depleted_color, dtype=np.uint8
        )

        # Draw grass first
        self._draw_grass(grid, simulation_state)

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

    def _draw_grass(self, grid: np.ndarray, sim_state: Any):
        """Draws the grass distribution on the grid."""
        resource_patches = sim_state.get_entities_with_components([ResourceComponent])
        env = sim_state.environment
        if not isinstance(env, CommonsEnvironment):
            return

        for entity_id, components in resource_patches.items():
            res_comp = cast(ResourceComponent, components.get(ResourceComponent))

            # Find the position from the resource_grid in the environment
            pos = None
            for p, eid in env.resource_grid.items():
                if eid == entity_id:
                    pos = p
                    break

            if pos:
                if res_comp.is_depleted:
                    color = self.depleted_color
                else:
                    ratio = res_comp.current_resource / res_comp.max_resource
                    color = (
                        self.grass_min_color * (1 - ratio)
                        + self.grass_max_color * ratio
                    )
                self._draw_pixel_block(grid, pos[0], pos[1], color.astype(np.uint8))

    def _draw_agents(self, grid: np.ndarray, sim_state: Any):
        """Draws agents on the grid."""
        entities = sim_state.get_entities_with_components([PositionComponent])
        for _, components in entities.items():
            pos_comp = cast(PositionComponent, components.get(PositionComponent))
            self._draw_pixel_block(grid, pos_comp.x, pos_comp.y, self.agent_color)
