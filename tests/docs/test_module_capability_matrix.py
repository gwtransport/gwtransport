"""Guard the ``modules.rst`` capability matrix against silent API drift.

The Forward/Inverse columns of the capability matrix in
``docs/source/user_guide/modules.rst`` are the only machine-checkable claim in
the module overview. This test pins them to the public API so that gaining or
losing a forward/inverse operation without updating the matrix fails CI.
"""

import importlib

import pytest

# Module -> (forward function name, inverse function name or None).
# Mirrors the Forward/Inverse columns of the modules.rst capability matrix.
EXPECTED = {
    "advection": ("infiltration_to_extraction", "extraction_to_infiltration"),
    "diffusion": ("infiltration_to_extraction", "extraction_to_infiltration"),
    "deposition": ("deposition_to_extraction", "extraction_to_deposition"),
    "recharge": ("recharge_to_extraction", None),
    "radial_asr": ("infiltration_to_extraction", "extraction_to_infiltration"),
}


@pytest.mark.parametrize(("module_name", "forward", "inverse"), [(m, f, i) for m, (f, i) in EXPECTED.items()])
def test_capability_matrix_forward_inverse(module_name, forward, inverse):
    """Forward exists for every row; inverse exists iff the matrix marks it."""
    module = importlib.import_module(f"gwtransport.{module_name}")

    assert callable(getattr(module, forward, None)), f"{module_name}.{forward} (Forward column) is missing"

    public_inverses = [
        name for name in vars(module) if name.startswith("extraction_to_") and callable(getattr(module, name))
    ]

    if inverse is None:
        assert not public_inverses, f"{module_name} has an inverse {public_inverses} but the matrix marks Inverse as –"
    else:
        assert callable(getattr(module, inverse, None)), f"{module_name}.{inverse} (Inverse column) is missing"
