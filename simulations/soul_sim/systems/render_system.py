# src/simulations/soul_sim/systems/render_system.py
"""
Handles the rendering of the simulation state to image frames for visualization.
"""

import os
from typing import List, Type, cast

import imageio.v2 as imageio
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from agent_core.core.ecs.component import Component, TimeBudgetComponent
from agent_engine.simulation.system import System

# Import world-specific components
from ..components import (
    InventoryComponent,
    NestComponent,
    PositionComponent,
    ResourceComponent,
)


class RenderSystem(System):
    """
    Renders the current grid state to a .png image for each simulation step.
    """

    REQUIRED_COMPONENTS: List[Type[Component]] = []  # Renders all entities

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use direct attribute access on the validated Pydantic model
        log_dir = self.config.simulation.log_directory
        self.image_frames_dir = os.path.join(log_dir, "frames")
        os.makedirs(self.image_frames_dir, exist_ok=True)
        self.frames_to_render: List[str] = []

    async def update(self, current_tick: int) -> None:
        """
        Renders the current grid state using the simulation_state object.
        """
        env = self.simulation_state.environment
        if not env or not hasattr(env, "width") or not hasattr(env, "height"):
            return

        fig, ax = self._setup_plot((env.width, env.height), current_tick)

        self._draw_resources(ax)
        self._draw_nests_and_farms(ax)
        self._draw_agents(ax)

        filepath = os.path.join(self.image_frames_dir, f"frame_{current_tick:04d}.png")
        plt.savefig(filepath, dpi=100)
        plt.close(fig)
        self.frames_to_render.append(filepath)

    def _setup_plot(self, grid_size: tuple, tick: int) -> tuple[plt.Figure, plt.Axes]:
        """Initializes the Matplotlib figure and axis."""
        fig, ax = plt.subplots(figsize=(grid_size[1] * 0.8, grid_size[0] * 0.8))
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, grid_size[1] - 0.5)
        ax.set_ylim(-0.5, grid_size[0] - 0.5)
        ax.set_xticks(np.arange(grid_size[1]))
        ax.set_yticks(np.arange(grid_size[0]))
        ax.grid(True, which="major", color="gray", linestyle="-", linewidth=0.5)
        ax.invert_yaxis()
        ax.set_title(f"Simulation Tick: {tick}")
        return fig, ax

    def _draw_resources(self, ax: plt.Axes):
        """Draws all resource nodes."""
        resource_colors = {
            "SINGLE_NODE": "green",
            "DOUBLE_NODE": "blue",
            "TRIPLE_NODE": "purple",
        }
        for _, comps in self.simulation_state.get_entities_with_components(
            [ResourceComponent, PositionComponent]
        ).items():
            res_comp = cast(ResourceComponent, comps.get(ResourceComponent))
            pos_comp = cast(PositionComponent, comps.get(PositionComponent))
            if not res_comp.is_depleted:
                color = resource_colors.get(res_comp.type, "gray")
                alpha = res_comp.current_health / res_comp.initial_health if res_comp.initial_health > 0 else 0.0
                rect = patches.Rectangle(
                    (pos_comp.position[1] - 0.25, pos_comp.position[0] - 0.25),
                    0.5,
                    0.5,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor="black",
                    zorder=2,
                )
                ax.add_patch(rect)

    def _draw_nests_and_farms(self, ax: plt.Axes):
        """Draws all nests and farms."""
        agent_entities = self.simulation_state.get_entities_with_components([TimeBudgetComponent])
        agent_ids = sorted(agent_entities.keys())
        if not agent_ids:
            return
        cmap = plt.get_cmap("hsv")

        for i, entity_id in enumerate(agent_ids):
            agent_color = cmap(i / len(agent_ids))
            components = agent_entities[entity_id]
            if isinstance(nest_comp := components.get(NestComponent), NestComponent):
                for nest_pos in nest_comp.locations:
                    ax.plot(
                        nest_pos[1],
                        nest_pos[0],
                        marker="H",
                        markersize=18,
                        color=agent_color,
                        markeredgecolor="black",
                        zorder=4,
                    )
            if isinstance(inv_comp := components.get(InventoryComponent), InventoryComponent) and inv_comp.farming_mode:
                if inv_comp.farm_location:
                    ax.plot(
                        inv_comp.farm_location[1],
                        inv_comp.farm_location[0],
                        marker="s",
                        markersize=15,
                        color=agent_color,
                        alpha=0.5,
                        zorder=3,
                    )

    def _draw_agents(self, ax: plt.Axes):
        """Draws all agents."""
        agent_entities = self.simulation_state.get_entities_with_components([TimeBudgetComponent, PositionComponent])
        agent_ids = sorted(agent_entities.keys())
        if not agent_ids:
            return
        cmap = plt.get_cmap("hsv")

        for i, entity_id in enumerate(agent_ids):
            components = agent_entities[entity_id]
            pos_comp = cast(PositionComponent, components.get(PositionComponent))
            time_comp = cast(TimeBudgetComponent, components.get(TimeBudgetComponent))
            agent_color = cmap(i / len(agent_ids))

            if time_comp.is_active:
                circle = patches.Circle(
                    (pos_comp.position[1], pos_comp.position[0]),
                    0.4,
                    facecolor=agent_color,
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=5,
                )
                ax.add_patch(circle)
                ax.text(
                    pos_comp.position[1],
                    pos_comp.position[0],
                    entity_id[-2:],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                    weight="bold",
                )
            else:
                ax.text(
                    pos_comp.position[1],
                    pos_comp.position[0],
                    "X",
                    color="red",
                    ha="center",
                    va="center",
                    fontsize=20,
                    weight="bold",
                    zorder=6,
                )

    def finalize(self):
        """Compiles the saved frames into a GIF at the end of the simulation."""
        if not self.frames_to_render:
            print("RenderSystem: No frames were rendered to create a GIF.")
            return

        log_dir = self.config.simulation.log_directory
        gif_path = os.path.join(log_dir, "simulation_output.gif")

        print(f"\nCreating simulation GIF from {len(self.frames_to_render)} frames...")
        try:
            images = [imageio.imread(filepath) for filepath in sorted(self.frames_to_render)]
            imageio.mimsave(gif_path, images, duration=200, loop=0)
            print(f"Simulation GIF saved to {gif_path}")
        except Exception as e:
            print(f"Error creating GIF: {e}")
