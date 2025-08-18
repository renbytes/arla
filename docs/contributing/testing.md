# Contributor's Guide: Running Tests

Testing is a critical part of the development process for the ARLA framework. We have a comprehensive test suite that includes unit, integration, and contract tests to ensure the reliability and correctness of the codebase.

All contributors are expected to run the test suite locally before submitting a pull request. Furthermore, all new features or bug fixes should be accompanied by corresponding tests.

## 1. Running the Full Test Suite

We've simplified the process of running tests with a single `Makefile` command. This command should be run from the root of the project.

```bash
make test
```

This command executes `pytest` inside the Docker container and does the following:

- Discovers and runs all tests in the `tests/` directory.
- Generates a code coverage report for the `agent-core` and `agent-engine` libraries.
- Fails the build if the total test coverage is below the configured threshold (currently 80%).

## 2. Continuous Integration (CI)

Our CI pipeline, powered by GitHub Actions, automatically runs the full test suite on every pull request and every push to the `main` branch.

A pull request **will not be merged** if any of the tests are failing or if the code coverage drops below the required threshold. Running the tests locally before you push your changes is the best way to ensure a smooth review process.

## 3. Writing New Tests

When you add a new feature or fix a bug, you should also add one or more tests to validate your changes.

- **Test Location**: Tests are organized in a parallel directory structure. For example, a test for a module located at `agent-engine/src/agent_engine/systems/reflection_system.py` should be placed at `tests/agent-engine/systems/test_reflection_system.py`.

- **Framework**: We use `pytest` as our testing framework. Please use `pytest`-style fixtures and assertion syntax.

- **Mocking**: We use `unittest.mock` for creating mock objects and patching dependencies to ensure tests are isolated.

Thank you for helping us maintain a high-quality and reliable codebase!
