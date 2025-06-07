"""gwtransport: A Python package for solving one-dimensional groundwater transport problems."""

import importlib.metadata as m

from gwtransport.utils import compute_time_edges

__version__ = m.version("gwtransport")
__all__ = ["compute_time_edges"]
