"""Configuration file for the Sphinx documentation builder."""  # noqa: INP001

import importlib.metadata
import json
import sys
import tomllib
import types
from datetime import date
from pathlib import Path

import nbformat

# PYODIDE_VERSION tracks the installed (unpinned) jupyterlite-pyodide-kernel; conf.py runs
# only under sphinx-build with the docs extras, so a direct import is safe and a future
# upstream rename fails loudly here instead of silently prefetching a stale Pyodide for the
# CDN prewarm. (ty lints conf.py without the docs extras; see [tool.ty.overrides].)
from jupyterlite_pyodide_kernel.constants import PYODIDE_VERSION

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
# The version is dynamic (hatch-vcs derives it from git tags), so it no longer
# appears in the [project] table; read it from the installed distribution instead.
release = importlib.metadata.version("gwtransport")

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
    # Must be listed after sphinx.ext.napoleon: its "autodoc-process-docstring" hook reads
    # napoleon's processed output to inject the "Try it live" button into Examples sections.
    "jupyterlite_sphinx",
]

# Napoleon settings for numpy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
# MUST stay False. With True, napoleon emits ".. admonition:: Examples" and indents the
# example body 3 spaces; jupyterlite-sphinx's try_examples parser only treats lines that
# literally start with ">>>" as code, so the indented block collapses into a single inert
# markdown cell instead of runnable code cells. Rubric output (False) keeps ">>>" at column
# 0 and converts correctly.
napoleon_use_admonition_for_examples = False
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
# "_interactive" holds the Pyodide-ready notebook copies generated below; "_contents" is
# where jupyterlite-sphinx stages notebooks for its build. Exclude both so nbsphinx does
# not also try to render them as standalone pages.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_interactive", "_contents"]

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

# numpy.typing.ArrayLike / NDArray expand to private ``numpy._typing`` internals (e.g.
# ``numpy._typing._array_like.GenericAlias`` / ``NDArray``) that are not exposed in the numpy
# intersphinx inventory; they render correctly but trip nitpicky mode, so ignore the namespace.
nitpick_ignore_regex = [
    ("py:class", r"numpy\._typing\..*"),
]

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for jupyterlite-sphinx (client-side interactive examples) -------
# Docstring "Examples" blocks (via try_examples) and the Pyodide-compatible example
# notebooks run entirely in the reader's browser through a JupyterLite/Pyodide kernel --
# no server. gwtransport is installed from the wheel published alongside these docs, so
# the interactive code runs the exact version the docs were built from (the dev version
# on the docs branch), not the last PyPI release.
jupyterlite_dev_wheel_url = (
    f"https://gwtransport.github.io/gwtransport/_static/wheels/gwtransport-{release}-py3-none-any.whl"
)

# Install gwtransport from the docs-site wheel (deps=False: numpy/scipy/pandas/matplotlib
# ship with Pyodide and requests is not needed client-side), then preload the
# Pyodide-native packages that gwtransport imports transitively.
_jupyterlite_install = (
    "import micropip\n"
    f"await micropip.install({jupyterlite_dev_wheel_url!r}, deps=False)\n"
    "import numpy, scipy, pandas, matplotlib, mpmath  # noqa: F401\n"
)

# Example notebooks that run client-side. Every notebook in ../examples is exposed as an
# interactive (JupyterLite/Pyodide) copy and picked up automatically -- adding a new example
# notebook requires no change here. Only the timflow notebook is excluded: it pulls in numba
# (transitively via timflow), which has no WebAssembly build.
_examples_dir = Path(__file__).parent.parent.parent / "examples"
_jupyterlite_excluded_notebooks = {"08_bank_filtration_timflow"}
jupyterlite_interactive_notebooks = sorted(
    nb.stem for nb in _examples_dir.glob("*.ipynb") if nb.stem not in _jupyterlite_excluded_notebooks
)

# nbsphinx already renders every .ipynb page; keep it that way (do not let
# jupyterlite-sphinx claim the .ipynb source suffix).
jupyterlite_bind_ipynb_suffix = False
jupyterlite_silence = True

# "Try it live" button on every NumPy-style Examples block.
global_enable_try_examples = True
try_examples_global_button_text = "Try it live"
# No warning banner cell at the top of the interactive frame (try_examples_global_warning_text
# defaults to None -> no banner). The Pyodide stack is still prefetched in the background; see
# the prewarm config below.
try_examples_preamble = _jupyterlite_install

# Background preloading: on pages that expose interactive examples, a small generated script
# (_static/js/pyodide-prewarm.js, written by setup() below) prefetches the Pyodide runtime and
# scientific-stack wheels from the CDN during browser idle time, so launching an example is
# near-instant. The CDN base is pinned to the Pyodide build shipped by jupyterlite-pyodide-kernel.
pyodide_cdn_base = f"https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/"
# Config (generated, sets window.gwtPrewarmConfig) must load before the prefetch logic.
html_js_files = ["js/pyodide-prewarm-config.js", "js/pyodide-prewarm.js"]

# jupyterlite-sphinx ships no CSS for the interactive-example buttons; this styles
# them to match the sphinx-book-theme (see _static/css/try-examples.css).
html_css_files = ["css/try-examples.css"]

# -- Options for nbsphinx ---------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"
# Jinja2 template (nbsphinx exposes ``env``). Every notebook keeps the location note;
# the Pyodide-compatible notebooks additionally get a button that opens their generated
# _interactive/ copy in a new browser tab (`:new_tab:`), rather than embedding a live
# iframe at the top of the static page. The button reuses jupyterlite-sphinx's
# `try_examples_button` class, so it matches the "Try it live" docstring buttons.
nbsphinx_prolog = (
    "{% set nbname = env.docname.split('/')|last %}\n"
    "\n"
    ".. note::\n"
    "   This notebook is located in the ./examples directory of the gwtransport repository.\n"
    "\n"
    "{% if nbname in " + repr(jupyterlite_interactive_notebooks) + " %}\n"
    ".. notebooklite:: /_interactive/{{ nbname }}.ipynb\n"
    "   :new_tab: true\n"
    "   :new_tab_button_text: Open this notebook live in a new tab\n"
    "{% endif %}\n"
)

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
    # The interactive-docs wheel is published into the site by the docs build itself, so
    # it does not exist at link-check time.
    r"https://gwtransport\.github\.io/gwtransport/_static/wheels/.*\.whl",
]


# -- Client-side interactive notebooks --------------------------------------
def _generate_interactive_notebooks(app):
    """Write Pyodide-ready copies of the launchable example notebooks.

    Each copy gets a leading cell that installs gwtransport from the docs-site wheel and
    preloads its Pyodide-native dependencies. The original notebooks -- rendered statically
    by nbsphinx and executed by the test suite -- are left untouched.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application, used to locate the source and ``examples`` directories.
    """
    examples_dir = Path(app.srcdir).parent.parent / "examples"
    out_dir = Path(app.srcdir) / "_interactive"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in jupyterlite_interactive_notebooks:
        nb = nbformat.read(examples_dir / f"{name}.ipynb", as_version=4)
        install_cell = nbformat.v4.new_code_cell(
            "# Auto-added for the in-browser (JupyterLite/Pyodide) build of this notebook.\n" + _jupyterlite_install
        )
        nb.cells.insert(0, install_cell)
        nbformat.write(nb, out_dir / f"{name}.ipynb")


def _write_prewarm_config(app):
    """Write the JS config consumed by ``pyodide-prewarm.js``.

    Injects the (version-pinned) Pyodide CDN base URL and the gwtransport wheel URL so the
    committed prefetch logic stays free of build-specific values.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application, used to locate the ``_static`` directory.
    """
    out_dir = Path(app.srcdir) / "_static" / "js"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"pyodideBase": pyodide_cdn_base, "gwtWheel": jupyterlite_dev_wheel_url}
    out_dir.joinpath("pyodide-prewarm-config.js").write_text(
        f"// Auto-generated by conf.py. Do not edit.\nwindow.gwtPrewarmConfig = {json.dumps(config)};\n",
        encoding="utf-8",
    )


def setup(app):
    """Register the build-time asset generators with Sphinx.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application to connect the ``builder-inited`` handlers to.
    """
    app.connect("builder-inited", _generate_interactive_notebooks)
    app.connect("builder-inited", _write_prewarm_config)
