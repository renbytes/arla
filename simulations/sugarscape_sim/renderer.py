# FILE: simulations/sugarscape_sim/renderer.py
import numpy as np
import imageio
from pathlib import Path
from typing import Any, cast

from .components import PositionComponent, EnergyComponent
from .environment import SugarscapeEnvironment


class SugarscapeRenderer:
    """Renders the state of the Sugarscape simulation grid to an image."""

    def __init__(self, width: int, height: int, output_dir: str, pixel_scale: int = 1):
        self.width = width
        self.height = height
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pixel_scale = pixel_scale

        # Define colors
        self.agent_color = np.array([236, 240, 241], dtype=np.uint8)  # White
        self.sugar_min_color = np.array([25, 25, 25], dtype=np.uint8)  # Dark Grey
        self.sugar_max_color = np.array([241, 196, 15], dtype=np.uint8)  # Yellow

    def render_frame(self, simulation_state: Any, tick: int) -> None:
        """Creates and saves a single frame of the simulation."""
        scaled_height = self.height * self.pixel_scale
        scaled_width = self.width * self.pixel_scale

        # Create a blank canvas
        grid = np.full(
            (scaled_height, scaled_width, 3), self.sugar_min_color, dtype=np.uint8
        )

        env = simulation_state.environment
        if not isinstance(env, SugarscapeEnvironment):
            return

        # Draw sugar map first
        self._draw_sugar(grid, env)

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

    def _draw_sugar(self, grid: np.ndarray, env: SugarscapeEnvironment):
        """Draws the sugar distribution on the grid."""
        for y in range(env.height):
            for x in range(env.width):
                sugar_level = env.get_sugar_at((x, y))
                if sugar_level > 0:
                    # Interpolate color based on sugar amount
                    ratio = sugar_level / env.max_sugar_per_cell
                    color = (
                        self.sugar_min_color * (1 - ratio)
                        + self.sugar_max_color * ratio
                    )
                    self._draw_pixel_block(grid, x, y, color.astype(np.uint8))

    def _draw_agents(self, grid: np.ndarray, sim_state: Any):
        """Draws agents on the grid."""
        entities = sim_state.get_entities_with_components(
            [PositionComponent, EnergyComponent]
        )
        for _, components in entities.items():
            pos_comp = cast(PositionComponent, components.get(PositionComponent))
            if pos_comp:
                self._draw_pixel_block(grid, pos_comp.x, pos_comp.y, self.agent_color)
