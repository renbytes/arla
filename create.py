import os
import pathlib


def create_project_structure(root_dir="agent-engine"):
    """
    Creates the complete directory and file structure for the agent-engine library.
    """
    # Define all directories to be created relative to the root
    dirs = [
        "src/agent_engine/simulation",
        "src/agent_engine/systems",
        "src/agent_engine/policy",
        "tests/systems",
    ]

    # Define all empty files to be created relative to the root
    files = [
        ".gitignore",
        "pyproject.toml",
        "README.md",
        "src/agent_engine/__init__.py",
        "src/agent_engine/simulation/__init__.py",
        "src/agent_engine/simulation/engine.py",
        "src/agent_engine/simulation/system.py",
        "src/agent_engine/systems/__init__.py",
        "src/agent_engine/systems/affect_system.py",
        "src/agent_engine/systems/causal_graph_system.py",
        "src/agent_engine/systems/goal_system.py",
        "src/agent_engine/systems/identity_system.py",
        "src/agent_engine/systems/q_learning_system.py",
        "src/agent_engine/systems/reflection_system.py",
        "src/agent_engine/policy/__init__.py",
        "src/agent_engine/policy/state_encoder.py",
        "src/agent_engine/policy/reward_calculator.py",
        "tests/__init__.py",
        "tests/systems/__init__.py",
        "tests/systems/test_affect_system.py",
    ]

    # Create the root directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        print(f"Created root directory: {root_dir}/")

    # Create all subdirectories
    for d in dirs:
        path = os.path.join(root_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}/")

    # Create all files
    for f in files:
        path = os.path.join(root_dir, f)
        # Use pathlib.Path.touch() to create an empty file
        pathlib.Path(path).touch()
        print(f"Created file:      {path}")


if __name__ == "__main__":
    create_project_structure()
    print("\nProject structure for agent-engine created successfully!")
    print("You can now begin migrating the system logic into these files.")
