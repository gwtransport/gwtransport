# gwtransport

Scientific Python package for timeseries analysis of groundwater transport of solutes and heat.

## Commands

```bash
# Setup (fresh environment, separate from user's .venv)
# Windows: replace `env VAR=val cmd` with `set "VAR=val" && cmd`
rm -rf .venv-claude
env UV_PROJECT_ENVIRONMENT=.venv-claude uv sync --all-extras -q
git config core.hooksPath .githooks               # Enable pre-commit hook

# Testing (run before committing)
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q pytest tests/src -n auto                   # Unit tests
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q pytest tests/examples -n auto              # Example notebooks
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q pytest tests/docs -n auto                  # Documentation code snippets

# Linting (run before committing)
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q ruff format .                              # Format code
env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q ruff check --fix .                         # Lint and auto-fix
npx prettier --check "**/*.{yaml,yml,md}"         # Format markdown/yaml

# Type checking (run before committing)
uv tool update -q ty & uv tool run -q ty check .

# Documentation
uv tool run -q --from sphinx --with-editable ".[docs]" sphinx-build -j auto -b linkcheck docs/source docs/build/linkcheck
rm -rf docs/build && uv tool run -q --from sphinx --with-editable ".[docs]" sphinx-build -j 1 -b html docs/source docs/build/html
```

## CI/CD

All checks must pass before merging. Pipeline tests on Python 3.12 (minimum deps) and 3.14 (latest deps). See `.github/workflows/` for details.

## Project Layout

- `src/gwtransport/` -- Package source code
- `tests/src/` -- Unit tests (one test file per module)
- `tests/examples/` -- Jupyter notebook execution tests
- `tests/docs/` -- Documentation code snippet tests
- `examples/` -- Example Jupyter notebooks
- `docs/source/` -- Sphinx documentation source

## Philosophy

You are a quality gatekeeper, not just an implementer. Before writing code:

- **Understand the physics.** This is a scientific package -- correctness of physical equations, units, and boundary conditions matters more than code elegance. If unsure about the physics, ask.
- **Check for dead code.** After every change, verify no unused imports, functions, or variables remain. Remove them.
- **Keep API and docs consistent.** When changing a public function signature, update its docstring, any cross-references (see `docs/CROSS_REFERENCES.md`), and affected example notebooks.
- **Re-read the request.** Before finishing, re-read the original question to verify you actually answered it.

## Code Style

- **Docstrings**: NumPy style. See example below.
- **Line length**: 120 characters.
- **Type hints**: Required for all public functions. Use `npt.ArrayLike` for array inputs, `npt.NDArray[np.floating]` for array outputs, `pd.DatetimeIndex` for time edges. Use built-in Python generics (`list`, `tuple`, `dict`, `X | None`) -- NEVER import from `typing`.
- **Vectorization**: ALWAYS prefer vectorized NumPy/SciPy/pandas operations over Python for-loops. If you find yourself writing a loop over array elements, stop and find the vectorized equivalent.
- **Formatting**: Enforced by linting with ruff and prettier. Do not fight the formatter.
- **Parameter names**: Use descriptive names consistent with domain conventions. For example, use `flow` for flow rate, `cin`/`cout` for concentrations, `tedges` for time edges, `xedges` for spatial edges.

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
    """
```

## Domain Conventions

**IMPORTANT**: These conventions are load-bearing for correctness.

- **Bin-edge pattern**: Time is represented as `tedges` (`pd.DatetimeIndex`, n+1 edges) with n values constant over each interval `[tedges[i], tedges[i+1])`. Same pattern for spatial dimension (`xedges`).
- **Input/output semantics**: Input values (`flow`, `cin` for infiltration-to-extraction; `flow`, `cout` for extraction-to-infiltration) are constant per bin. Output concentration/temperature is a flow-weighted bin average.
- **Paired operations**: Functions come in forward (infiltration-to-extraction) and reverse (extraction-to-infiltration) variants.
- **Multiple parameterizations**: Gamma distributions support both (alpha, beta, loc) and (mean, std, loc).
- **Retardation**: All transport functions account for sorption via retardation factors.
- **Units**: Must be consistent within a calculation. The package does not enforce units -- the user is responsible.

## Testing

- Use fixtures from `tests/src/conftest.py` for common test data.
- Tests MUST be exact to machine precision. Use `np.testing.assert_allclose(actual, expected)`.
- Validate physical correctness: conservation laws, boundary conditions, limiting cases.
- Tests MUST be meaningful -- not trivial identity checks. Use analytical solutions for validation when possible.
- Run specific tests with: `uv run pytest tests/src/test_diffusion.py -v` or `uv run pytest tests/src -k "advection" -v`
- LLM reviewers (including this one) routinely flag plausible-sounding bugs that do not actually exist; before applying any non-trivial suggested change, write a regression test for the claimed bug and run it against the unchanged baseline -- only proceed if the baseline test fails.

## Git

- Do NOT include Claude-related signatures in commit messages or PR descriptions.
- Base your PR message on the template at `.github/pull_request_template.md`.
- Run formatting, linting, and type checking before committing.

## Cross-References

See `docs/CROSS_REFERENCES.md` for available Sphinx labels (concepts, assumptions, examples) and syntax for linking from docstrings vs notebooks.
