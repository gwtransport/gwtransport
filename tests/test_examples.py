import subprocess
import sys


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
    result = subprocess.run([sys.executable, pythonfile_path], capture_output=True, text=True, check=True)
    if result.returncode != 0:
        msg = f"Failed executing {pythonfile_path}.\n\n{result.stderr}"
        raise AssertionError(msg)
    return result.stdout
