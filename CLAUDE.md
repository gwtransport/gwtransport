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
npx prettier --check "**/*.{yaml,yml,md}"  # Format markdown

# Type checking
uv tool update ty
uv tool run ty check .

# Build documentation
uv tool run --from sphinx --with-editable ".[docs]" sphinx-build -j auto -b linkcheck docs/source docs/build/linkcheck # Check links in documentation
uv tool run --from sphinx --with-editable ".[docs]" sphinx-build -j auto -b html docs/source docs/build/html # Build HTML documentation
```

## CI/CD Pipeline

All checks must pass before merging. The pipeline tests on Python 3.11 (minimum deps) and 3.14 (latest deps).

| Check          | Command                                     | Workflow               |
| -------------- | ------------------------------------------- | ---------------------- |
| Formatting     | `uv run ruff format --diff .`               | linting.yml            |
| Linting        | `uv run ruff check .`                       | linting.yml            |
| Type checking  | `uv run ty check .`                         | linting.yml            |
| pyproject.toml | `uv run validate-pyproject pyproject.toml`  | linting.yml            |
| YAML/Markdown  | `npx prettier --check "**/*.{yaml,yml,md}"` | linting.yml            |
| Unit tests     | `uv run pytest tests/src`                   | functional_testing.yml |
| Example tests  | `uv run pytest tests/examples`              | examples_testing.yml   |
| Doc tests      | `uv run pytest tests/docs`                  | docs.yml               |
| Link check     | `sphinx-build -b linkcheck docs/source ...` | docs.yml               |

## Code Style

- **Docstrings**: NumPy style (enforced via ruff)
- **Line length**: 120 characters
- **Type hints**: Required for all public functions
- **Formatting**: `ruff format` with preview mode enabled

Example docstring (use `npt.ArrayLike` for array inputs, `pd.DatetimeIndex` for time edges):

```python
def function_name(*, flow: npt.ArrayLike, tedges: pd.DatetimeIndex) -> npt.NDArray[np.floating]:
    """Short description.

    Parameters
    ----------
    flow : array-like
        Flow rate (m³/day).
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n values).

    Returns
    -------
    ndarray
        Result description.

    See Also
    --------
    related_function : Brief description.
    :ref:`concept-residence-time` : Background on the concept.
    :ref:`assumption-linear-retardation` : When this assumption applies.
    """
```

## Architecture

```
src/gwtransport/
├── advection.py          # Main advection transport with pore volume distributions
├── diffusion.py          # 1D advection-dispersion analytical solutions (slow, physically rigorous)
├── diffusion_fast.py     # Diffusive corrections via Gaussian smoothing (fast, approximate)
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
uv run pytest tests/src/test_diffusion.py -v    # Single module
uv run pytest tests/src -k "advection" -v        # By keyword
uv run pytest tests/src --cov=src                # With coverage
```

**Writing tests**:

- Use fixtures from `tests/src/conftest.py` for common test data
- Tests should be exact to machine precision. Use `np.testing.assert_allclose(actual, expected)` for numerical comparisons.
- Validate physical correctness (conservation, bounds, limiting cases)
- Tests and comparisons should be meaningful and not trivial
- Use analytical solutions for validation when possible

## Git

- Do not include Claude-related signatures in commit messages
- Run `ruff format .`, `ruff check --fix .`, `uv tool update ty` and `uv tool run ty check .` before committing

## Conventions and Customs

```python
tedges = pd.DatetimeIndex([...])  # n+1 edges
values = np.array([...])           # n values
```

- Values represent **average** over interval
- Interval: `[tedges[i], tedges[i+1])`
- Same holds for the spatial dimension: xedges

Units must be consistent within calculation

## Documentation Cross-References

Enrich function docstrings with references to concepts and assumptions when they meaningfully aid understanding. Use Sphinx cross-references that render as clickable links.

**Syntax by context**:

| Context                | Syntax                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------- |
| Python docstrings      | `:ref:`concept-dispersion-scales``                                                            |
| Jupyter notebooks / md | `[Dispersion scales](https://gwtransport.github.io/gwtransport/user_guide/concepts.html#...)` |
| API links (notebooks)  | `[module](https://gwtransport.github.io/gwtransport/api/modules.html#module-gwtransport.xxx)` |

Base URL: `https://gwtransport.github.io/gwtransport/`

**Available labels** (in `docs/source/user_guide/`):

_Concepts_ (`concepts.rst` → `concepts.html`):

| Label                              | Topic                                    |
| ---------------------------------- | ---------------------------------------- |
| `concept-pore-volume-distribution` | Central concept: aquifer heterogeneity   |
| `concept-residence-time`           | Time in aquifer (V·R/Q)                  |
| `concept-retardation-factor`       | Slower movement due to sorption          |
| `concept-transport-equation`       | Flow-weighted averaging                  |
| `concept-dispersion`               | Macroscopic spreading from heterogeneity |
| `concept-dispersion-scales`        | Scale-dependent heterogeneity            |
| `concept-gamma-distribution`       | Two-parameter pore volume model          |
| `concept-nonlinear-sorption`       | Freundlich isotherm, front-tracking      |

_Assumptions_ (`assumptions.rst` → `assumptions.html`):

| Label                             | Topic                        |
| --------------------------------- | ---------------------------- |
| `assumption-advection-dominated`  | When diffusion is negligible |
| `assumption-steady-streamlines`   | Fixed flow path geometry     |
| `assumption-gamma-distribution`   | Gamma distribution adequacy  |
| `assumption-linear-retardation`   | Constant retardation factor  |
| `assumption-no-reactions`         | Conservative transport       |
| `assumption-no-transverse-mixing` | Independent streamtubes      |
| `assumptions`                     | Full assumptions page        |

_Examples_ (`../examples/` → `examples.html`):

| Path                                               | Topic                               |
| -------------------------------------------------- | ----------------------------------- |
| `examples/01_Aquifer_Characterization_Temperature` | Temperature tracer calibration      |
| `examples/02_Residence_Time_Analysis`              | Residence time calculations         |
| `examples/03_Pathogen_Removal_Bank_Filtration`     | Log removal efficiency              |
| `examples/04_Deposition_Analysis_Bank_Filtration`  | Deposition analysis                 |
| `examples/05_Diffusion_Dispersion`                 | Dispersion vs APVD, equivalent std  |
| `examples/10_Advection_with_non_linear_sorption`   | Freundlich sorption, front-tracking |

**When to add references**: Function assumes something non-obvious, user needs context to choose between functions, or physical limitations affect interpretation.

**Keep it minimal** - only add references that genuinely help users understand when/how to use a function
