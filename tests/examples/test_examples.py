import os
import subprocess
import sys
import tempfile

import nbformat
import pytest
from nbconvert import PythonExporter  # type: ignore[attr-defined]


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
    env = os.environ.copy()
    # Force a non-interactive matplotlib backend to avoid blocking GUI calls
    env.setdefault("MPLBACKEND", "Agg")
    # Run with a timeout to avoid hanging interactive prompts
    result = subprocess.run(
        [sys.executable, pythonfile_path], capture_output=True, text=True, check=False, env=env, timeout=30
    )
    if result.returncode != 0:
        msg = f"Script {pythonfile_path} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        raise AssertionError(msg)


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

    # Sanitize converted body: remove IPython magics and input() calls that would block
    sanitized_lines = []
    for src_line in body.splitlines():
        stripped = src_line.strip()
        # Skip IPython magic commands and shell escapes
        if stripped.startswith(("%", "!")):
            continue
        # Replace input() calls with a noop to avoid blocking
    safe_line = src_line.replace("input(", "lambda *args, **kw: None(") if "input(" in src_line else src_line
    sanitized_lines.append(safe_line)

    sanitized_body = "\n".join(sanitized_lines)

    # Prepend a safety header: disable input() and force non-interactive plotting
    safety_header = "import builtins\nbuiltins.input = lambda *a, **k: None\nimport matplotlib\nmatplotlib.use('Agg')\n"

    # Create a temporary Python file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", encoding="utf-8") as temp_file:
        temp_file.write(safety_header + sanitized_body)
        temp_file.flush()
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

    # The notebook contains unexecuted cells; ensure that is detected
    with pytest.raises(AssertionError):
        test_notebook_cells_executed(failing_notebook_path)
