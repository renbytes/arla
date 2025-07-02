# Contributing to the ARLA Dynamics Platform

First off, thank you for considering contributing! This project aims to be a collaborative platform for exploring and comparing different models of artificial cognition. Your contributions are invaluable.

This document provides a set of guidelines for contributing to the ARLA Dynamics Platform. These are mostly guidelines, not strict rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under Issues.

If you're unable to find an open issue addressing the problem, open a new one. Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for a new feature or a change to an existing one, please start by opening an issue. This allows for discussion before you invest a significant amount of time in development.

When creating an enhancement suggestion, please include:

- A clear and descriptive title to identify the suggestion
- A step-by-step description of the suggested enhancement in as much detail as possible
- Specific examples to demonstrate the steps. Include copy/pasteable snippets which you use in those examples, as appropriate
- Explain why this enhancement would be useful to most ARLA Dynamics users

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking through `good-first-issue` and `help-wanted` issues:

- **Good first issues** - issues which should only require a few lines of code, and a test or two
- **Help wanted issues** - issues which should be a bit more involved than good first issues

## Development Process

### Automated Code Style & Quality Checks

To maintain a high-quality and consistent codebase, this project uses pre-commit hooks. These are automated checks that run on your code before you are allowed to finalize a commit.

When you run `git commit`, the following tools will automatically check the files you've staged:

- **Ruff:** An extremely fast Python linter and code formatter. It will check for style issues and automatically reformat your code to match the project's standards
- **Mypy:** A static type checker. It will analyze your type hints to catch potential bugs before they become runtime errors
- **File Cleanliness Hooks:** Other hooks will automatically fix common issues like trailing whitespace and ensure files end with a single newline

### Your Workflow:

1. Write your code and stage your files with `git add`
2. Run `git commit -m "Your message"` ([good site here](https://www.conventionalcommits.org/en/v1.0.0/) on good commit message conventions)
3. The pre-commit hooks will now run
4. If the hooks fail, your commit will be aborted. Many issues (like formatting) will be fixed for you automatically. Simply review the changes, `git add` the modified files again, and re-run your commit command
5. If the hooks pass, your commit will be created successfully

This automated process ensures that all code entering the repository adheres to the same standards, making it easier for everyone to read, review, and maintain.

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Install dependencies and set up your development environment. Make sure to run `pre-commit install` to set up the hooks locally
3. Make your changes. Please adhere to the coding style enforced by the pre-commit hooks
4. Add or update tests for your changes. Your PR will not be accepted if it decreases test coverage
5. Update the documentation. If you are adding a new component, system, or changing behavior, please update the relevant documentation in the `/docs` directory
6. Ensure the test suite passes locally before submitting your PR
7. Submit your Pull Request. Please link the PR to the issue it resolves

For a detailed guide on the project's architecture and how to add new cognitive modules, please see our [Developer Guide](DEVELOPER_GUIDE.md).
