# FILE: agent-sim/src/agent_sim/infrastructure/ray_config.py
"""
Centralized configuration and initialization for the Ray cluster.
"""

import ray
from rich import print


def init_ray():
    """
    Initializes the Ray cluster.

    This function checks if Ray is already connected and, if not,
    initializes a new Ray instance. This allows for a flexible setup,
    supporting both local development and connection to a remote cluster.
    """
    if ray.is_initialized():
        print("✅ Ray is already initialized.")
        return

    try:
        print("🚀 Initializing Ray cluster...")
        # This will start a new Ray instance locally if one is not found.
        # For production, this could be modified to connect to a specific address.
        ray.init(address="auto", ignore_reinit_error=True)
        print("[bold green]✔ Ray cluster initialized successfully.[/bold green]")
        print(
            f"   - Dashboard URL: [link={ray.get_dashboard_url()}]http://127.0.0.1:8265[/link]"
        )
    except Exception as e:
        print(f"[bold red]❌ Failed to initialize Ray cluster: {e}[/bold red]")
        print("   - Please ensure Ray is properly installed.")
        raise
