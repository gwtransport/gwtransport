# gwtransport

Scientific Python package for timeseries analysis of groundwater transport of solutes and heat.

## Quick Reference

```bash
# Setup
uv sync --all-extras

# Testing (run before committing)
uv run pytest tests/src                    # Unit tests
uv run pytest tests/examples               # Example notebooks
uv run pytest tests/docs                   # Documentation code snippets

# Linting (run before committing)
uv run ruff format .                       # Format code
uv run ruff check --fix .                  # Lint and auto-fix

# Type checking
uv tool update ty
uv tool run ty check .

# Build documentation
uv tool run --from sphinx --with-editable ".[docs]" sphinx-build -j auto -b html docs/source docs/build/html
```

## CI/CD Pipeline

All checks must pass before merging. The pipeline tests on Python 3.11 (minimum deps) and 3.14 (latest deps).

| Check          | Command                                      | Workflow               |
| -------------- | -------------------------------------------- | ---------------------- |
| Formatting     | `uv run ruff format --diff .`                | linting.yml            |
| Linting        | `uv run ruff check .`                        | linting.yml            |
| Type checking  | `uv run ty check .`                          | linting.yml            |
| pyproject.toml | `uv run validate-pyproject pyproject.toml`   | linting.yml            |
| YAML/Markdown  | `npx prettier --check "**/*.{yaml,yml,md}"`  | linting.yml            |
| Unit tests     | `uv run pytest tests/src`                    | functional_testing.yml |
| Example tests  | `uv run pytest tests/examples`               | examples_testing.yml   |
| Doc tests      | `uv run pytest tests/docs`                   | docs.yml               |
| Link check     | `sphinx-build -b linkcheck docs/source ...`  | docs.yml               |

## Code Style

- **Docstrings**: NumPy style (enforced via ruff)
- **Line length**: 120 characters
- **Type hints**: Required for all public functions
- **Formatting**: `ruff format` with preview mode enabled

Example docstring:

```python
def function_name(param1: float, param2: np.ndarray) -> np.ndarray:
    """Short description of function.

    Longer description if needed.

    Parameters
    ----------
    param1 : float
        Description of param1.
    param2 : ndarray
        Description of param2.

    Returns
    -------
    ndarray
        Description of return value.

    Examples
    --------
    >>> result = function_name(1.0, np.array([1, 2, 3]))
    """
```

## Architecture

```
src/gwtransport/
├── advection.py          # Main advection transport with pore volume distributions
├── diffusion2.py         # 1D advection-dispersion analytical solutions
├── diffusion.py          # Diffusive corrections via Gaussian smoothing
├── residence_time.py     # Residence time calculations with retardation
├── deposition.py         # Deposition process analysis
├── logremoval.py         # Log removal efficiency calculations
├── gamma.py              # Gamma distribution utilities for pore volumes
├── utils.py              # General utilities (interpolation, bin operations)
└── fronttracking/        # Event-driven solver for nonlinear sorption
    ├── solver.py         # Main simulation engine
    ├── output.py         # Result extraction and formatting
    ├── handlers.py       # Event handlers
    ├── math.py           # Shock/rarefaction wave calculations
    ├── waves.py          # Wave structure definitions
    ├── events.py         # Event type definitions
    ├── plot.py           # Visualization tools
    └── validation.py     # Physics validation
```

**Key patterns**:

- **Paired operations**: Forward (infiltration→extraction) and reverse (extraction→infiltration)
- **Multiple parameterizations**: Support both (alpha, beta) and (mean, std) for distributions
- **Retardation support**: All transport functions account for sorption

## Testing

Tests are organized in `tests/`:

- `tests/src/` - Unit tests for each module
- `tests/examples/` - Jupyter notebook execution tests
- `tests/docs/` - Documentation code snippet tests

**Running specific tests**:

```bash
uv run pytest tests/src/test_diffusion2.py -v    # Single module
uv run pytest tests/src -k "advection" -v        # By keyword
uv run pytest tests/src --cov=src                # With coverage
```

**Writing tests**:

- Use fixtures from `tests/src/conftest.py` for common test data
- Tests should be exact to machine precision. Use `np.testing.assert_allclose(actual, expected)` for numerical comparisons.
- Validate physical correctness (conservation, bounds, limiting cases)
- Tests should be meaningful and not trivial
- Use analytical solutions for validation when possible

## Git

- Do not include Claude-related signatures in commit messages
- Run `ruff format .`, `ruff check --fix .`, `uv tool update ty` and `uv tool run ty check .` before committing
