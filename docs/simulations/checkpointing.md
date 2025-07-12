# Simulation Management Guide

This guide covers how to run, stop, and restore simulations in the ARLA framework.

## 1. How to Run a New Simulation

This process involves starting the necessary background services (the message broker and Celery workers) and then submitting the experiment definition file.

### Prerequisites

You must have a message broker, like Redis, running. If you're using Docker, you can start one with:

```bash
docker run -d -p 6379:6379 redis
```

### Steps

1. **Open a new terminal** and start a Celery worker dedicated to running the actual simulations. This worker consumes from the `simulations` queue.

   ```bash
   agentsim start-worker --queue simulations
   ```

2. **Open a second terminal** and start a Celery worker that orchestrates the experiments. This worker consumes from the `experiments` queue.

   ```bash
   agentsim start-worker --queue experiments
   ```

3. **Open a third terminal** to submit your experiment. Point the command to your experiment definition file from the project root (`arla/`).

   ```bash
   agentsim run-experiment agent-sim/experiments/test_ablation.yml
   ```

The orchestrator will read the file, create all the variations, and queue them for the simulation workers.

## 2. How to Gracefully Stop a Simulation

To stop the simulation environment, you must shut down the Celery workers.

1. Navigate to the terminals where your two Celery workers are running.
2. Press `Ctrl+C` in each terminal.

Celery will perform a "warm shutdown." It will finish any simulation runs that are currently in progress but will not start any new ones from the queue.

## 3. How to Restore a Simulation from a Checkpoint

The CLI is designed to run new experiments. To restore a specific run from a checkpoint, you'll need to use a dedicated script.

### Steps

1. **Locate Your Checkpoint File**. Checkpoints are saved automatically in the logs directory, with a path similar to this:

   ```
   data/logs/snapshots/sim_1721334681_a4c1b3f2/snapshot_tick_150.json
   ```

2. **Create a Restore Script**. Create a new Python file in your project, for example, `scripts/restore_run.py`.

3. **Add the Following Code**. This script manually calls the `setup_and_run` function, passing in the path to your checkpoint.

   ```python
   # scripts/restore_run.py
   import asyncio
   from omegaconf import OmegaConf
   from simulations.soul_sim.run import setup_and_run

   # --- CONFIGURATION ---
   # 1. Paste the path to your saved checkpoint file here
   CHECKPOINT_PATH = "data/logs/snapshots/sim_1721334681_a4c1b3f2/snapshot_tick_150.json"

   # 2. Provide IDs for the new, restored run
   RUN_ID = "restored_run_01"
   TASK_ID = "manual_restore_01"
   EXPERIMENT_ID = "restored_experiment"


   async def main():
       """
       Loads a base configuration, then runs the simulation starting
       from the specified checkpoint.
       """
       print(f"--- Restoring simulation from: {CHECKPOINT_PATH} ---")

       # Load the base configuration file.
       base_config = OmegaConf.load("simulations/soul_sim/config/base_config.yml")

       # The setup_and_run function handles the entire lifecycle
       await setup_and_run(
           run_id=RUN_ID,
           task_id=TASK_ID,
           experiment_id=EXPERIMENT_ID,
           config_overrides=base_config,
           checkpoint_path=CHECKPOINT_PATH,
       )

   if __name__ == "__main__":
       asyncio.run(main())
   ```

4. **Run the Script**. Execute your new script from the terminal to start the simulation from the specified file.

   ```bash
   python scripts/restore_run.py
   ```

## Additional Notes

### Terminal Management

For better organization, consider using terminal multiplexers like `tmux` or `screen` to manage multiple workers:

```bash
# Using tmux
tmux new-session -d -s arla-workers
tmux new-window -t arla-workers -n simulations 'agentsim start-worker --queue simulations'
tmux new-window -t arla-workers -n experiments 'agentsim start-worker --queue experiments'
tmux attach -t arla-workers
```

### Monitoring

You can monitor your Celery workers and queues using:

```bash
# Check worker status
celery -A agent_sim.infrastructure.tasks.celery_app inspect active

# Monitor in real-time (if Flower is installed)
celery -A agent_sim.infrastructure.tasks.celery_app flower
```

### Troubleshooting

**Common Issues:**

- **Redis not running**: Ensure Redis is started before launching workers
- **Import errors**: Make sure all packages are installed with `python install.py`
- **Permission errors**: Check that the data/logs directory is writable
- **Worker hangs**: Use `Ctrl+C` followed by `kill -9 <pid>` if needed

For more detailed logs, add `--loglevel=debug` to your worker commands.
