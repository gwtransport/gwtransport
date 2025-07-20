"""Configuration file for the Sphinx documentation builder."""  # noqa: INP001

import tomllib
from pathlib import Path

# Read project information from pyproject.toml
pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project_info = pyproject_data["project"]

# -- Project information -----------------------------------------------------
project = project_info["name"]
copyright = "2025, Bas des Tombe"  # noqa: A001
author = ", ".join([author["name"] for author in project_info["authors"]])
release = project_info["version"]

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx_copybutton",
]

# Napoleon settings for numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Templates path
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "gwtransport"
html_static_path = ["_static"]

# Sphinx Book Theme options
html_theme_options = {
    "repository_url": "https://github.com/gwtransport/gwtransport",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 3,
    "navigation_with_keys": True,
    "show_navbar_depth": 2,
    "path_to_docs": "docs/source",
    "launch_buttons": {
        "notebook_interface": "classic",
        "binderhub_url": "",
        "jupyterhub_url": "",
        "thebe": False,
        "colab_url": "",
    },
}

# -- Options for autodoc ----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Handle type annotation compatibility
autodoc_type_aliases = {
    "npt.NDArray": "numpy.ndarray",
    "Sequence": "typing.Sequence",
}

# Mock imports for compatibility - empty for now to test actual imports
autodoc_mock_imports = []

# -- Options for intersphinx extension --------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for nbsphinx ---------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"

# Set the path for notebooks
nbsphinx_prolog = """
.. note::
   This notebook is located in the ./examples directory of the gwtransport repository.
"""

# -- Options for copybutton -------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
