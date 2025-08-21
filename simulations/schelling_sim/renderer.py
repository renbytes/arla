# simulations/schelling_sim/renderer.py

import numpy as np
import imageio
from pathlib import Path
from typing import Any

from .components import PositionComponent, GroupComponent


class SchellingRenderer:
    """Renders the state of the Schelling simulation grid to an image."""

    def __init__(self, width: int, height: int, output_dir: str, pixel_scale: int = 1):
        self.width = width
        self.height = height
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pixel_scale = pixel_scale

        self.colors = {
            0: [25, 25, 25],
            1: [52, 152, 219],
            2: [231, 76, 60],
        }

    def render_frame(self, simulation_state: Any, tick: int) -> None:
        """Creates and saves a single frame of the simulation."""
        # Create a larger grid based on the scale
        scaled_height = self.height * self.pixel_scale
        scaled_width = self.width * self.pixel_scale
        grid = np.full((scaled_height, scaled_width, 3), self.colors[0], dtype=np.uint8)

        entities = simulation_state.get_entities_with_components(
            [PositionComponent, GroupComponent]
        )

        for _, components in entities.items():
            pos_comp = components.get(PositionComponent)
            group_comp = components.get(GroupComponent)

            if pos_comp and group_comp:
                # Color a block of pixels instead of a single one
                y_start = pos_comp.y * self.pixel_scale
                y_end = y_start + self.pixel_scale
                x_start = pos_comp.x * self.pixel_scale
                x_end = x_start + self.pixel_scale

                # Use array slicing to set the color for the entire block
                grid[y_start:y_end, x_start:x_end] = self.colors[group_comp.agent_type]

        frame_path = self.output_path / f"frame_{tick:04d}.png"
        imageio.imwrite(frame_path, grid)
