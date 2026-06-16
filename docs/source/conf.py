"""Configuration file for the Sphinx documentation builder."""  # noqa: INP001

import sys
import tomllib
import types
from datetime import date
from pathlib import Path

# nbsphinx-link 1.3.1 imports SafeString/ErrorString from
# docutils.utils.error_reporting, which was removed in docutils 0.22 (required
# by Sphinx 9). Those were Python 2 unicode-coercion helpers; str() is a safe
# Python 3 substitute. Remove once nbsphinx-link ships a fix for
# https://github.com/vidartf/nbsphinx-link/issues/25.
if "docutils.utils.error_reporting" not in sys.modules:
    _shim = types.ModuleType("docutils.utils.error_reporting")
    setattr(_shim, "SafeString", str)  # noqa: B010
    setattr(_shim, "ErrorString", str)  # noqa: B010
    sys.modules["docutils.utils.error_reporting"] = _shim

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
    "array_like": ":py:data:`~numpy.typing.ArrayLike`",
    "callable": ":py:class:`~collections.abc.Callable`",
    "ndarray": ":py:class:`~numpy.ndarray`",
    "DatetimeIndex": ":py:class:`~pandas.DatetimeIndex`",
    # Internal fronttracking types
    "Wave": ":py:class:`~gwtransport.fronttracking.waves.Wave`",
    "ShockWave": ":py:class:`~gwtransport.fronttracking.waves.ShockWave`",
    "CharacteristicWave": ":py:class:`~gwtransport.fronttracking.waves.CharacteristicWave`",
    "RarefactionWave": ":py:class:`~gwtransport.fronttracking.waves.RarefactionWave`",
    "FreundlichSorption": ":py:class:`~gwtransport.fronttracking.math.FreundlichSorption`",
    "ConstantRetardation": ":py:class:`~gwtransport.fronttracking.math.ConstantRetardation`",
    "SorptionModel": ":py:data:`~gwtransport.fronttracking.math.SorptionModel`",
    "FrontTrackerState": ":py:class:`~gwtransport.fronttracking.solver.FrontTrackerState`",
    "Event": ":py:class:`~gwtransport.fronttracking.events.Event`",
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
    "show_toc_level": 2,
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

# nbsphinx sets a callable in nbsphinx_custom_formats which cannot be pickled
# for Sphinx's configuration cache; the warning is benign.
suppress_warnings = ["config.cache"]

# numpy.float64 is emitted by autodoc when rendering the signatures of
# functions that accept ``npt.NDArray[np.float64]`` but is not exposed as a
# py:class in the numpy intersphinx inventory.
nitpick_ignore = [
    ("py:class", "numpy.float64"),
]

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

# -- Options for linkcheck --------------------------------------------------
# The coverage badge and HTML coverage report are build artifacts published to the site by the
# coverage-deploy step; they exist only after a successful deploy, so a pre-deploy linkcheck (or a
# run whose coverage deploy lagged behind a red test run) sees a 404. They are self-referential
# build outputs, not stable external URLs to validate, so exclude them from linkcheck.
linkcheck_ignore = [
    r"https://gwtransport\.github\.io/gwtransport/coverage-badge\.svg",
    r"https://gwtransport\.github\.io/gwtransport/htmlcov/?",
]
