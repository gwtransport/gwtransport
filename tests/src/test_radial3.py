"""Tests for the analytical-kernel radial push-pull module.

The module under test is :mod:`gwtransport.radial3`, which evaluates a
closed-form 1-D erf-in-volume-coordinate advection-diffusion kernel on
a Gauss-Legendre partition of the union of all bin edges. The tests
exercise:

1. **Pure advection** -- with ``D_m == 0`` and ``alpha_L == 0`` the
   path delegates to the LIFO matrix in
   :mod:`gwtransport.radial_utils`, so it must match to machine
   precision.
2. **Structural invariants** -- constant ``cin == c_background`` must
   give constant ``cout == c_background``; row-stochastic structure.
3. **Arbitrary ``cout_tedges``** -- the output grid can be shifted or
   refined relative to ``tedges`` and the bin averages must match.
4. **Inverse round-trip** -- forward then inverse recovers ``cin``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from gwtransport.radial3 import (
    gamma_push_pull,
    gamma_push_pull_inverse,
    push_pull,
    push_pull_inverse,
)

_T0 = pd.Timestamp("2024-01-01")


def _build_simple_push_pull(n_inject: int = 15, n_extract: int = 15, q: float = 100.0):
    """Symmetric push-pull: inject at ``+q`` then extract at ``-q``."""
    n = n_inject + n_extract
    tedges = pd.date_range(_T0, periods=n + 1, freq="1D")
    flow = np.concatenate([q * np.ones(n_inject), -q * np.ones(n_extract)])
    cin = np.concatenate([np.ones(n_inject), np.zeros(n_extract)])
    return flow, cin, tedges


def _build_alternating(n_cycles: int = 5, burst: int = 3, q: float = 100.0):
    """Alternating inject/extract cycles, constant ``cin=1`` during inject."""
    cycle_flow = np.concatenate([q * np.ones(burst), -q * np.ones(burst)])
    cycle_cin = np.concatenate([np.ones(burst), np.zeros(burst)])
    flow = np.tile(cycle_flow, n_cycles)
    cin = np.tile(cycle_cin, n_cycles)
    n = len(flow)
    tedges = pd.date_range(_T0, periods=n + 1, freq="1D")
    return flow, cin, tedges


class TestPureAdvection:
    """``D_m == 0 and alpha_L == 0`` -- must match the LIFO fast path."""

    def test_symmetric_push_pull_constant_cin(self):
        flow, cin, tedges = _build_simple_push_pull()
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=0.0,
            c_background=0.0,
        )
        finite = np.isfinite(c)
        assert finite.sum() == 15
        assert_allclose(c[finite], 1.0, atol=1e-12)

    def test_alternating_flow_constant_cin(self):
        flow, cin, tedges = _build_alternating(n_cycles=4, burst=3, q=100.0)
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=0.0,
            c_background=0.0,
        )
        finite = np.isfinite(c)
        assert finite.sum() == 12  # 4 cycles x 3 extract days each
        assert_allclose(c[finite], 1.0, atol=1e-12)


class TestStructuralInvariants:
    """Conservation properties of the analytical kernel."""

    def test_uniform_field_preserved_with_diffusion(self):
        """``cin == c_background`` implies ``cout == c_background``."""
        flow, _, tedges = _build_simple_push_pull()
        cin = 0.5 * np.ones_like(flow)
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            c_background=0.5,
            molecular_diffusivity=0.02,
        )
        finite = np.isfinite(c)
        assert_allclose(c[finite], 0.5, atol=1e-10)

    def test_uniform_field_preserved_alternating(self):
        """Constant ``cin == c_background`` across alternating bursts."""
        flow, cin, tedges = _build_alternating(n_cycles=5, burst=3, q=100.0)
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            c_background=1.0,
            molecular_diffusivity=0.02,
        )
        finite = np.isfinite(c)
        assert_allclose(c[finite], 1.0, atol=1e-10)

    def test_uniform_field_preserved_with_dispersion(self):
        """Dispersion-only run with constant cin must preserve the field."""
        flow, _, tedges = _build_simple_push_pull()
        cin = 0.7 * np.ones_like(flow)
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            c_background=0.7,
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.5,
        )
        finite = np.isfinite(c)
        assert_allclose(c[finite], 0.7, atol=1e-10)


class TestArbitraryCoutTedges:
    """Output grid can differ from the input grid."""

    def test_refined_cout_grid_preserves_bin_average(self):
        flow, cin, tedges = _build_simple_push_pull()
        cout_tedges_fine = pd.date_range(_T0, periods=len(flow) * 2 + 1, freq="12h")
        c_fine = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges_fine,
            layer_heights=[5.0],
            porosity=0.3,
            c_background=0.0,
            molecular_diffusivity=0.02,
        )
        c_coarse = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            c_background=0.0,
            molecular_diffusivity=0.02,
        )
        for i in np.flatnonzero(np.isfinite(c_coarse)):
            sub = c_fine[2 * i : 2 * i + 2]
            if not np.all(np.isfinite(sub)):
                continue
            assert_allclose(sub.mean(), c_coarse[i], atol=5e-3)


class TestInverseRecovery:
    """Inverse path recovers ``cin`` from a forward-synthesized ``cout``."""

    def test_pure_advection_round_trip(self):
        flow, cin, tedges = _build_simple_push_pull()
        cout = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=0.0,
            c_background=0.0,
        )
        cin_rec = push_pull_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=0.0,
            c_background=0.0,
            regularization_strength=1e-6,
        )
        inj = flow > 0
        assert_allclose(cin_rec[inj], cin[inj], atol=1e-6)


class TestGammaWrappers:
    """Thin gamma-distribution wrappers must run and return sane shapes."""

    def test_gamma_push_pull_runs(self):
        flow, cin, tedges = _build_simple_push_pull()
        c = gamma_push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=8,
            porosity=0.3,
            c_background=0.0,
        )
        assert c.shape == (len(flow),)

    def test_gamma_push_pull_inverse_runs(self):
        flow, cin, tedges = _build_simple_push_pull()
        cout = gamma_push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=8,
            porosity=0.3,
            c_background=0.0,
        )
        cin_rec = gamma_push_pull_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=8,
            porosity=0.3,
            c_background=0.0,
            regularization_strength=1e-6,
        )
        assert cin_rec.shape == (len(flow),)
