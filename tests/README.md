# Tests Directory

This directory contains all unit and integration tests for the ADAN trading bot. Tests are crucial for ensuring the correctness and reliability of the codebase.

## Important Subdirectories:

- `unit/`: Contains unit tests for individual modules and functions. These tests focus on isolated components to verify their behavior.
- `integration/`: Contains integration tests that verify the interaction between multiple components or the end-to-end flow of certain functionalities.

## Usage:

To run all unit tests, navigate to the project root and execute:

```bash
python -m unittest discover tests/unit
```

To run all integration tests:

```bash
python -m unittest discover tests/integration
```

To run all tests (unit and integration):

```bash
python -m unittest discover tests
```

## Related Documentation:

- [Main README.md](../../README.md)