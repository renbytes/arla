# Agent Concurrent

**A lightweight, decoupled library for concurrent and parallel execution of agent-based simulation systems.**

`agent-concurrent` provides a simple and effective way to manage the execution flow of systems within a simulation. It offers both serial and asynchronous runners, allowing developers to choose the best execution strategy for their needs—whether for debugging, strict sequential logic, or high-performance parallel processing.

---

### Key Features

- **Decoupled Design:** Uses Python's `Protocol` to define the expected system interface, meaning it has no direct dependency on `agent-core` or `agent-engine`. It can run any list of objects that have an `async def update()` method.
- **Asynchronous Execution:** The `AsyncSystemRunner` leverages Python's `asyncio` library to run all system updates concurrently, significantly speeding up simulations where systems are independent.
- **Serial Execution:** The `SerialSystemRunner` executes systems one by one in a defined order, which is ideal for debugging or when a strict order of operations is required.
- **Robust Error Handling:** Both runners are designed to gracefully handle exceptions within a system, logging the error without crashing the entire simulation loop.

---

### Installation

You can install the library directly from PyPI using pip:

```bash
pip install agent-concurrent
```

## Development Setup

To set up a local development environment, clone the repository and install it in editable mode with its development dependencies.

1. **Clone the repository:**

```bash
git clone https://github.com/renbytes/agent-concurrent.git
cd agent-concurrent
```

2. **Create and activate a virtual environment (recommended):**

```bash
conda create --name agent-concurrent python=3.11
conda activate agent-concurrent
```

3. **Install the package in editable mode with dev dependencies:** This command installs the package and includes tools like `pytest` for testing and `mypy` for static type checking.

```bash
pip install -e ".[dev]"
```

## Basic Usage

The library is designed to be straightforward to use. You simply choose a runner and pass it a list of system objects to execute.

```python
import asyncio
from agent_concurrent import AsyncSystemRunner

# Define systems that conform to the SystemProtocol
# (i.e., they have an async update method)
class MySystemA:
    async def update(self, current_tick: int):
        print(f"System A updating for tick {current_tick}...")
        await asyncio.sleep(0.1)

class MySystemB:
    async def update(self, current_tick: int):
        print(f"System B updating for tick {current_tick}...")
        await asyncio.sleep(0.1)

async def main():
    # Instantiate the runner
    runner = AsyncSystemRunner()

    # Create a list of systems
    systems_to_run = [MySystemA(), MySystemB()]

    # Execute the systems for a given tick
    await runner.run(systems=systems_to_run, current_tick=1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Running Tests and Type Checking

To ensure code quality and correctness, you can run the included unit tests and static type checker.

1. **Run Unit Tests:** From the root of the project directory, run `pytest`:

```bash
pytest
```

2. **Run Static Type Checker:** To check for type errors with `mypy`, run:

```bash
mypy src
```

3. **Install Pre-Commit Hooks:** To check errors prior to pushing code, run:

```bash
pre-commit install
```
