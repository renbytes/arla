

```
celery -A agent-sim.src.agent_sim.infrastructure.tasks.celery_app worker --loglevel=INFO -Q simulations
```

```
python scripts/task_manager.py experiment \
  --package=soul_sim \
  --scenarios=src/simulations/soul_sim/scenarios/default.json \
  --runs-per-scenario=1
```
