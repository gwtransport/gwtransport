"""Pytest configuration file."""

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
