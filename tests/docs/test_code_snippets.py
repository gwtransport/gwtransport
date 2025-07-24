"""Tests for code snippets found in documentation files."""

import re
from pathlib import Path


def _get_all_code_snippets():
    """Get all Python code snippets from RST files and README.md."""
    repo_root = Path(__file__).parent.parent.parent
    docs_dir = repo_root / "docs" / "source"
    snippets = []

    # Scan RST files in docs/source
    for rst_file in docs_dir.glob("*.rst"):
        content = rst_file.read_text(encoding="utf-8")
        # RST format: .. code-block:: python
        for match in re.finditer(
            r"^\.\. code(?:-block)?:: python\s*\n((?:^\s*\n|^[ ]{3}.*\n)*)", content, re.MULTILINE
        ):
            code = re.sub(r"^[ ]{3}", "", match.group(1), flags=re.MULTILINE).strip()
            if code:
                line_num = content[: match.start()].count("\n") + 1
                snippets.append((code, line_num, rst_file.name))

    # Scan README.md
    readme_file = repo_root / "README.md"
    if readme_file.exists():
        content = readme_file.read_text(encoding="utf-8")
        # Markdown format: ```python
        for match in re.finditer(r"^```python\s*\n(.*?)^```", content, re.MULTILINE | re.DOTALL):
            code = match.group(1).strip()
            if code:
                line_num = content[: match.start()].count("\n") + 1
                snippets.append((code, line_num, "README.md"))

    return snippets


def pytest_generate_tests(metafunc):
    """Generate tests for each code snippet found."""
    if "code_snippet_data" in metafunc.fixturenames:
        snippets = _get_all_code_snippets()
        metafunc.parametrize("code_snippet_data", snippets, ids=[f"{fname}:{line}" for _, line, fname in snippets])


def test_code_snippet(code_snippet_data):
    """Test that documentation code snippets execute without error."""
    code, line_num, file_name = code_snippet_data
    exec_globals = {"np": __import__("numpy"), "pd": __import__("pandas")}

    try:
        exec(code, exec_globals)  # noqa: S102
    except Exception as e:
        msg = f"{file_name}:{line_num} failed: {e}"
        raise AssertionError(msg) from e
