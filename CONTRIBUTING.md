# Contributing to RoboQA-Temporal

Thank you for your interest in contributing to RoboQA-Temporal! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/roboqa-temporal.git`
3. Create a virtual environment: `python3 -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux)
5. Source ROS: `source /opt/ros/humble/setup.bash` (Linux)
6. Install dependencies: `pip install -r requirements.txt`
7. Install in development mode: `pip install -e .`

## Development Workflow

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Make your changes
3. Write or update tests
4. Ensure all tests pass: `pytest tests/`
5. Commit your changes: `git commit -m "feat/docs/fix/test/revert: Description of changes"`
6. Push to your fork: `git push origin feat/your-feature-name`
7. Open a Pull Request

## Guidelines & Suggestions:

### Code Style

- Follow standard python coding style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

### Testing

- Write tests for new features
- Ensure existing tests still pass
- Aim for good test coverage
- Use descriptive test names

### Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Update examples if API changes
- Keep configuration examples up to date

### Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure CI checks pass
- Request review from maintainers

### Reporting Issues

When reporting bugs or requesting features:

- Provide a clear description
- Include steps to reproduce (for bugs)
- Include relevant system information
- Provide example data if possible (anonymized)

## Questions?

Feel free to open an issue for questions or discussions about contributions.

Thank you for contributing to RoboQA-Temporal!

