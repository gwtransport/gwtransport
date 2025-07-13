import os
import subprocess
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


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
    Execute a Jupyter notebook as a test using nbconvert.

    Parameters
    ----------
    ipynb_path : str
        The path to the Jupyter notebook to execute.

    Raises
    ------
    AssertionError
        If the Jupyter notebook does not execute successfully.
    """
    with open(ipynb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Get the directory containing the notebook for proper working directory
    notebook_dir = os.path.dirname(os.path.abspath(ipynb_path))

    # Extract kernel name from notebook metadata, fallback to python3
    kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")

    # Configure the preprocessor with appropriate settings for testing
    ep = ExecutePreprocessor(
        timeout=300,  # 5 minutes timeout per cell
        kernel_name=kernel_name,
        allow_errors=False,  # Fail fast on any cell error
        record_timing=False,  # Don't need timing info for tests
    )

    try:
        # Execute the notebook in its own directory
        ep.preprocess(nb, {"metadata": {"path": notebook_dir}})
    except Exception as e:
        # Re-raise with more context about which notebook failed
        error_msg = f"Notebook execution failed for {ipynb_path}: {e}"
        raise AssertionError(error_msg) from e
