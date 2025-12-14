"""Configuration file for the Sphinx documentation builder."""  # noqa: INP001

import tomllib
from datetime import date
from pathlib import Path

# Read project information from pyproject.toml
pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

project_info = pyproject_data["project"]

# -- Project information -----------------------------------------------------
project = project_info["name"]
copyright = f"{date.today().year}, Bas des Tombe"  # noqa: A001, DTZ011
author = ", ".join([author["name"] for author in project_info["authors"]])
release = project_info["version"]

# -- General configuration ---------------------------------------------------
extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinxext.opengraph",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Napoleon settings for numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "array-like": ":py:data:`~numpy.typing.ArrayLike`",
    "callable": ":py:class:`~collections.abc.Callable`",
}

templates_path = ["_templates"]
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

# Prevent type aliases from being expanded into their full definitions
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "NDArray": "NDArray",
}

# sphinx-autodoc-typehints configuration
typehints_fully_qualified = False
always_use_bars_union = True
typehints_defaults = "comma"
simplify_optional_unions = True
nitpicky = True

# -- Options for intersphinx -------------------------------------------------
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
nbsphinx_prolog = """
.. note::
   This notebook is located in the ./examples directory of the gwtransport repository.
"""

# -- Options for copybutton -------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for OpenGraph --------------------------------------------------
ogp_site_url = "https://gwtransport.github.io/gwtransport/"
ogp_description_length = 300
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="keywords" content="groundwater, transport, solutes, temperature, residence times, pathogen removal, timeseries analysis">',
]
