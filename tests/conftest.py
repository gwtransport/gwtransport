"""Pytest configuration file."""

import builtins
import os
from glob import glob


def pytest_generate_tests(metafunc):
    """
    Generate tests dynamically based on the presence of the "pythonfile_path" fixture.

    If "pythonfile_path" is among the fixture names requested by a test, this function
    discovers all Python files under the "examples" directory (including subdirectories),
    sorts the file paths, and then parametrizes the test function with each discovered
    Python file path.

    Parameters
    ----------
    metafunc : _pytest.python.Metafunc
        The test context object used by pytest to create function arguments and
        perform test setup.
    """
    if "pythonfile_path" in metafunc.fixturenames:
        filepaths = sorted(glob(os.path.join("examples", "*.py")))
        metafunc.parametrize("pythonfile_path", filepaths)

    if "ipynb_path" in metafunc.fixturenames:
        filepaths = sorted(glob(os.path.join("examples", "*.ipynb")))
        metafunc.parametrize("ipynb_path", filepaths)


def pytest_configure(config):  # noqa: ARG001
    """Global test configuration: make plotting non-interactive and disable input()."""
    # Force non-interactive matplotlib backend for tests
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Use platformdirs for jupyter to avoid deprecation warnings being raised as errors
    os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

    # Disable interactive input to prevent tests from hanging
    builtins.input = lambda _: None  # type: ignore[assignment]
