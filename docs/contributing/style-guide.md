# Contributor's Guide: Coding Standards & Style

To maintain a high level of code quality and consistency across the ARLA project, we adhere to a set of coding standards and architectural patterns. All contributions should follow these guidelines.

Our CI pipeline automatically checks for compliance, so following these rules will help ensure your pull requests are merged smoothly.

## 1. Formatting & Linting with Ruff

We use **Ruff** as an all-in-one tool for linting and code formatting. It's incredibly fast and helps us enforce a consistent style based on the PEP 8 standard.

- **Formatting**: Before committing your code, please run the formatter from the root of the project.

```bash
poetry run ruff format .
```

- **Linting**: To check for potential errors or style violations, run the linter.

```bash
poetry run ruff check .
```

Our CI pipeline will fail if either of these checks does not pass, so it's a good practice to run them locally before pushing your changes.

## 2. Static Type Checking with Mypy

We use **Mypy** to enforce static type checking. All new code should include type hints for function arguments, return values, and variables. This helps us catch bugs before they happen and makes the codebase easier to understand and maintain.

You can run the type checker locally with the following command:

```bash
poetry run mypy .
```

## 3. Architectural Principles

- **Separation of Concerns**: This is the most important principle in the ARLA framework.
  - **Components** must only contain data (state). They should not have any logic.
  - **Systems** must only contain logic. They should not have any local state beyond what's necessary for their operation.
  - **Actions** define what is possible but delegate the implementation of the logic to Systems via the Event Bus.

- **Dependency Injection**: Systems and other core classes receive their dependencies (like the `SimulationState` or configuration objects) through their constructors. Avoid using global objects or singletons.

- **Event-Driven Communication**: Systems must not call each other directly. All inter-system communication must happen through the **Event Bus**.

## 4. Docstrings and Comments

- **Docstrings**: All public modules, classes, and functions should have a clear, concise docstring explaining their purpose, arguments, and what they return. We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings.

- **Comments**: Use comments sparingly. Your code should be as self-documenting as possible through clear variable names and well-structured logic.
