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
        assert_allclose(c[finite], 0.5, atol=1e-13)

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
        assert_allclose(c[finite], 1.0, atol=1e-13)

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
        assert_allclose(c[finite], 0.7, atol=1e-13)


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


class TestBoundaryInvariants:
    """Qualitative physics checks on push-rest-pull cases.

    These verify monotonicity and sign of the diffusion response,
    without hardcoding a specific reference value. A quantitative
    comparison to an analytical GF at ``r=0`` is intentionally *not*
    done here -- see ``docs/research/radial3-vtau-cutoff.md`` for why
    a naive ``1 - exp(-V_max/(4*scale*D*t))`` expression is not the
    right reference (it assumes instantaneous push emplacement and
    point evaluation, whereas the kernel delivers a flow-weighted bin
    average over a finite extract interval).
    """

    @staticmethod
    def _push_rest_pull(
        n_push: int = 10,
        n_rest: int = 10,
        n_extract: int = 10,
        q: float = 100.0,
    ):
        flow = np.concatenate([q * np.ones(n_push), np.zeros(n_rest), -q * np.ones(n_extract)])
        cin = np.concatenate([np.ones(n_push), np.zeros(n_rest), np.zeros(n_extract)])
        tedges = pd.date_range(_T0, periods=len(flow) + 1, freq="1D")
        return flow, cin, tedges, n_push, n_rest, n_extract

    def test_cout_equals_cin_last_at_first_extract_zero_diffusion(self):
        """Zero diffusion: first extract bin must equal ``cin[last_inject]``."""
        flow, cin, tedges, n_push, n_rest, _ = self._push_rest_pull()
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
        first_extract = n_push + n_rest
        assert_allclose(c[first_extract], 1.0, atol=1e-12)

    def test_diffusion_reduces_cout_at_first_extract(self):
        """Large ``D_m`` plus long rest must drive ``cout[first_extract]`` below ``0.5``.

        Uses ``D_m = 10 m^2/day`` and ``R_rest = 10 days`` with the
        standard push setup. Analytic 2-D radial GF prediction is
        ``c_out ~ 0.412``, so the bin-average must lie strictly below
        ``0.5``. A delta-kernel bug that produces ``cout = 1`` will
        fail.
        """
        flow, cin, tedges, n_push, n_rest, _ = self._push_rest_pull(n_rest=10)
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=10.0,
            c_background=0.0,
        )
        first_extract = n_push + n_rest
        c_first = c[first_extract]
        assert np.isfinite(c_first)
        assert c_first < 0.5, f"expected diffusion to reduce cout[first] below 0.5, got {c_first}"

    def test_cout_first_extract_monotone_in_diffusivity(self):
        """``cout[first_extract]`` must strictly decrease as ``D_m`` grows."""
        flow, cin, tedges, n_push, n_rest, _ = self._push_rest_pull(n_rest=10)
        first_extract = n_push + n_rest
        d_values = [0.1, 1.0, 10.0, 100.0]
        cout_first = []
        for d_m in d_values:
            c = push_pull(
                flow=flow,
                cin=cin,
                tedges=tedges,
                cout_tedges=tedges,
                layer_heights=[5.0],
                porosity=0.3,
                molecular_diffusivity=d_m,
                c_background=0.0,
            )
            cout_first.append(c[first_extract])
        cout_first = np.asarray(cout_first)
        assert np.all(np.isfinite(cout_first))
        assert np.all(np.diff(cout_first) < -1e-6), f"expected strict decrease, got {cout_first}"
        assert cout_first[-1] < 0.1, f"expected near-zero cout at high D_m, got {cout_first[-1]}"

    def test_cout_monotone_non_increasing_during_extraction(self):
        """Across the extraction run, ``cout`` must be monotone non-increasing."""
        flow, cin, tedges, n_push, n_rest, _ = self._push_rest_pull(n_rest=10)
        c = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=10.0,
            c_background=0.0,
        )
        first_extract = n_push + n_rest
        c_ext = c[first_extract:]
        assert np.all(np.isfinite(c_ext))
        assert np.all(np.diff(c_ext) <= 1e-10), f"not monotone: {c_ext}"


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


class TestMachinePrecisionIdentities:
    """Algebraic identities of the forward operator ``cout = W @ cin_clean``.

    These tests depend only on row-sum structure and linearity, so
    they must hold to machine precision regardless of the physics
    correctness of ``W``. A regression that breaks any of them
    indicates a bookkeeping bug (row-sum drift, background-slack bug,
    non-linear contamination) rather than a physics modeling choice.
    """

    @staticmethod
    def _base_kwargs(tedges):
        return {
            "tedges": tedges,
            "cout_tedges": tedges,
            "layer_heights": [5.0],
            "porosity": 0.3,
            "molecular_diffusivity": 0.02,
        }

    def test_constant_field_preserved_to_machine_precision(self):
        """``cin == c_background`` must give ``cout == c_background`` to ~1e-14."""
        flow, _, tedges = _build_alternating(n_cycles=5, burst=3, q=100.0)
        c_bg = 0.37
        c = push_pull(
            flow=flow,
            cin=c_bg * np.ones_like(flow),
            c_background=c_bg,
            **self._base_kwargs(tedges),
        )
        finite = np.isfinite(c)
        assert_allclose(c[finite], c_bg, atol=1e-14)

    def test_affine_shift_invariance(self):
        """``cout(cin + k, c_bg + k) == cout(cin, c_bg) + k`` exactly."""
        flow, cin, tedges = _build_simple_push_pull()
        k = 0.73
        c_bg = 0.21
        c0 = push_pull(flow=flow, cin=cin, c_background=c_bg, **self._base_kwargs(tedges))
        c1 = push_pull(
            flow=flow,
            cin=cin + k,
            c_background=c_bg + k,
            **self._base_kwargs(tedges),
        )
        finite = np.isfinite(c0) & np.isfinite(c1)
        assert_allclose(c1[finite], c0[finite] + k, atol=1e-14)

    def test_linearity_in_cin_with_zero_background(self):
        """With ``c_background == 0`` the operator is linear in ``cin``."""
        flow, cin_a, tedges = _build_simple_push_pull()
        rng = np.random.default_rng(0)
        cin_b = rng.uniform(0.0, 1.0, size=cin_a.shape)
        alpha, beta = 0.4, 1.7
        kw = self._base_kwargs(tedges) | {"c_background": 0.0}
        c_a = push_pull(flow=flow, cin=cin_a, **kw)
        c_b = push_pull(flow=flow, cin=cin_b, **kw)
        c_mix = push_pull(flow=flow, cin=alpha * cin_a + beta * cin_b, **kw)
        finite = np.isfinite(c_a) & np.isfinite(c_b) & np.isfinite(c_mix)
        assert_allclose(c_mix[finite], alpha * c_a[finite] + beta * c_b[finite], atol=1e-14)

    def test_zero_diffusion_refined_cout_grid_exact(self):
        """With ``D_m == 0`` refined and coarse ``cout_tedges`` must agree exactly."""
        flow, cin, tedges = _build_simple_push_pull()
        cout_tedges_fine = pd.date_range(_T0, periods=len(flow) * 2 + 1, freq="12h")
        common = {
            "flow": flow,
            "cin": cin,
            "tedges": tedges,
            "layer_heights": [5.0],
            "porosity": 0.3,
            "molecular_diffusivity": 0.0,
            "c_background": 0.0,
        }
        c_fine = push_pull(cout_tedges=cout_tedges_fine, **common)
        c_coarse = push_pull(cout_tedges=tedges, **common)
        for i in np.flatnonzero(np.isfinite(c_coarse)):
            sub = c_fine[2 * i : 2 * i + 2]
            if np.all(np.isfinite(sub)):
                assert_allclose(sub.mean(), c_coarse[i], atol=1e-13)

    def test_zero_injection_returns_background(self):
        """All-extract flow with no history must give ``cout == c_background``."""
        n = 10
        tedges = pd.date_range(_T0, periods=n + 1, freq="1D")
        flow = -50.0 * np.ones(n)
        cin = np.zeros(n)
        c_bg = 0.42
        c = push_pull(flow=flow, cin=cin, c_background=c_bg, **self._base_kwargs(tedges))
        finite = np.isfinite(c)
        assert finite.any()
        assert_allclose(c[finite], c_bg, atol=1e-14)

    def test_porosity_layer_height_product_invariance(self):
        """``scale = n_layers * pi * h * n * R`` enters only as a product.

        Halving ``layer_height`` and doubling ``porosity`` must leave
        ``cout`` unchanged to machine precision. A bug that threads
        either parameter into a second code path (e.g., advection
        velocity) will break this identity.
        """
        flow, cin, tedges = _build_simple_push_pull()
        c_a = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[5.0],
            porosity=0.3,
            molecular_diffusivity=0.02,
            c_background=0.0,
        )
        c_b = push_pull(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=[2.5],
            porosity=0.6,
            molecular_diffusivity=0.02,
            c_background=0.0,
        )
        finite = np.isfinite(c_a) & np.isfinite(c_b)
        assert_allclose(c_a[finite], c_b[finite], atol=1e-14)


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
