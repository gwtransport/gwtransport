import os
import subprocess
import sys
import tempfile

import nbformat
import pytest
from nbconvert import PythonExporter


def test_pythonscript(pythonfile_path):
    """
    Execute a Python script as a test.

    Parameters
    ----------
    pythonfile_path : str
        The path to the Python script to execute.

    Raises
    ------
    AssertionError
        If the Python script does not execute successfully.
    """
    result = subprocess.run([sys.executable, pythonfile_path], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_ipynb(ipynb_path):
    """
    Convert a Jupyter notebook to a Python script and execute it as a test.

    Instead of using nbclient or nbconvert, so that coverage works correctly.

    Parameters
    ----------
    ipynb_path : str
        The path to the Jupyter notebook to convert and execute.

    Raises
    ------
    AssertionError
        If the converted Python script does not execute successfully.
    """
    # Read the notebook
    with open(ipynb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # Convert notebook to Python script
    exporter = PythonExporter()
    body = exporter.from_notebook_node(nb)[0]

    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as temp_file:
        temp_file.write(body)
        test_pythonscript(temp_file.name)


def test_notebook_cells_executed(ipynb_path):
    """
    Test that notebook cells have been executed by checking execution numbers.

    Parameters
    ----------
    ipynb_path : str
        The path to the Jupyter notebook to check.

    Raises
    ------
    AssertionError
        If any code cell lacks an execution number.
    """
    with open(ipynb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            execution_count = cell.get("execution_count")
            assert execution_count is not None, (
                f"Cell {i} in {ipynb_path} has not been executed (execution_count is None)"
            )
            assert isinstance(execution_count, int), (
                f"Cell {i} in {ipynb_path} has invalid execution_count: {execution_count}"
            )


@pytest.mark.xfail(reason="This notebook is expected to fail during execution")
def test_failing_notebook_xfail():
    """
    Test that the failing notebook fails as expected.

    This test is marked with xfail since the notebook is designed to fail.
    """
    failing_notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "failing_notebook.ipynb")

    # Verify the failing notebook exists
    assert os.path.exists(failing_notebook_path), f"Test notebook not found: {failing_notebook_path}"

    # This should fail due to unexecuted cells
    test_notebook_cells_executed(failing_notebook_path)


def test_failing_notebook_detection():
    """
    Test that failing notebooks are properly detected and raise AssertionError.

    This test ensures our notebook testing infrastructure correctly catches
    execution failures in notebooks.
    """
    failing_notebook_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "failing_notebook.ipynb")

    # Verify the failing notebook exists
    assert os.path.exists(failing_notebook_path), f"Test notebook not found: {failing_notebook_path}"

    # The test_ipynb function should raise AssertionError for failing notebooks
    with pytest.raises(AssertionError, match="Notebook execution failed"):
        test_ipynb(failing_notebook_path)
