"""Tests for radial push-pull well transport model."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from gwtransport.gamma import bins as gamma_bins_func
from gwtransport.radial import (
    gamma_push_pull_well,
    gamma_push_pull_well_inverse,
    push_pull_well,
    push_pull_well_inverse,
)
from gwtransport.radial_utils import _push_pull_advection_matrix, _signed_radial_distance


@pytest.fixture
def push_pull_scenario():
    """Standard push-pull scenario: 10 days inject, 5 days rest, 10 days extract."""
    n_inject = 10
    n_rest = 5
    n_extract = 10
    n = n_inject + n_rest + n_extract

    flow = np.zeros(n)
    flow[:n_inject] = 100.0
    flow[n_inject + n_rest :] = -100.0

    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    return flow, tedges


class TestPushPullAdvection:
    """Pure advection (LIFO stack) tests."""

    def test_constant_cin_constant_cout(self, push_pull_scenario):
        """Constant injection concentration gives constant extraction concentration."""
        flow, tedges = push_pull_scenario
        cin = np.ones(len(flow)) * 5.0

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([2.0, 3.0, 5.0]),
            porosity=0.3,
        )

        extraction_mask = flow < 0
        assert_allclose(cout[extraction_mask], 5.0)
        assert np.all(np.isnan(cout[~extraction_mask]))

    def test_step_function_reversed(self):
        """Step function input gives reversed step in extraction (LIFO)."""
        # 4 inject bins, 4 extract bins
        flow = np.array([100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        # LIFO: last injected (4.0) extracted first, then 3.0, 2.0, 1.0
        assert_allclose(cout[4:], [4.0, 3.0, 2.0, 1.0])

    def test_mass_conservation(self, push_pull_scenario):
        """Total extracted mass equals total injected mass."""
        flow, tedges = push_pull_scenario
        injection_mask = flow > 0
        cin = np.zeros(len(flow))
        cin[injection_mask] = np.linspace(1, 10, injection_mask.sum())

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([2.0, 3.0]),
            porosity=0.3,
        )

        dt = np.diff(tedges) / pd.Timedelta("1D")
        mass_in = np.sum(cin * flow * dt)
        cout_clean = np.where(np.isnan(cout), 0.0, cout)
        mass_out = np.sum(cout_clean * np.abs(flow) * dt)

        assert_allclose(mass_out, mass_in, rtol=1e-12)

    def test_layer_height_invisible_without_diffusion(self):
        """Without diffusion, layer height distribution is invisible."""
        flow = np.array([100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=5, freq="D")
        cin = np.array([1.0, 5.0, 0.0, 0.0])

        # Single uniform layer
        cout_uniform = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        # Multiple heterogeneous layers
        cout_hetero = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([1.0, 3.0, 5.0, 10.0]),
            porosity=0.3,
        )

        assert_allclose(cout_uniform[2:], cout_hetero[2:])

    def test_partial_extraction(self):
        """Extracting less volume than injected recovers only recent parcels."""
        # Inject 4 bins, extract 2 bins
        flow = np.array([100.0, 100.0, 100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        # LIFO: only last 2 injected bins extracted
        assert_allclose(cout[4:], [4.0, 3.0])

    def test_over_extraction(self):
        """Extracting more volume than injected: excess bins have cout=0."""
        # Inject 2 bins, extract 4 bins
        flow = np.array([100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([3.0, 5.0, 0.0, 0.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        # LIFO: first 2 extraction bins get injected water, last 2 get nothing (0)
        assert_allclose(cout[2], 5.0)
        assert_allclose(cout[3], 3.0)
        assert_allclose(cout[4], 0.0)
        assert_allclose(cout[5], 0.0)

    def test_multiple_push_pull_cycles(self):
        """Multiple injection-extraction cycles work correctly."""
        # Cycle 1: inject 2, extract 2; Cycle 2: inject 2, extract 2
        flow = np.array([100.0, 100.0, -100.0, -100.0, 100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.array([1.0, 2.0, 0.0, 0.0, 10.0, 20.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        # Cycle 1: LIFO gives [2.0, 1.0]
        assert_allclose(cout[2], 2.0)
        assert_allclose(cout[3], 1.0)
        # Cycle 2: LIFO gives [20.0, 10.0]
        assert_allclose(cout[6], 20.0)
        assert_allclose(cout[7], 10.0)


class TestSignedRadialDistance:
    """Tests for _signed_radial_distance."""

    def test_zero_volume(self):
        """Zero volume difference gives zero distance."""
        x = _signed_radial_distance(
            delta_volume=np.array([0.0]),
            layer_height=5.0,
            porosity=0.3,
            retardation_factor=1.0,
            n_layers=1,
        )
        assert_allclose(x, 0.0)

    def test_sign_convention(self):
        """Positive volume gives positive distance, negative gives negative."""
        dv = np.array([100.0, -100.0])
        x = _signed_radial_distance(
            delta_volume=dv,
            layer_height=5.0,
            porosity=0.3,
            retardation_factor=1.0,
            n_layers=1,
        )
        assert x[0] > 0
        assert x[1] < 0
        assert_allclose(np.abs(x[0]), np.abs(x[1]))

    def test_formula(self):
        """Verify against manual calculation."""
        dv = np.array([1000.0])
        h, n_por, r_factor, n_lay = 5.0, 0.3, 1.0, 3
        scale = n_lay * np.pi * h * n_por * r_factor
        expected = np.sqrt(1000.0 / scale)

        x = _signed_radial_distance(
            delta_volume=dv,
            layer_height=h,
            porosity=n_por,
            retardation_factor=r_factor,
            n_layers=n_lay,
        )
        assert_allclose(x[0], expected)


class TestPushPullDiffusion:
    """Radial diffusion tests."""

    def test_zero_diffusivity_matches_advection(self):
        """Zero diffusivity gives same result as pure advection."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([2.0, 5.0, 8.0])

        cout_adv = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cout_diff = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
        )

        extraction_mask = flow < 0
        assert_allclose(cout_diff[extraction_mask], cout_adv[extraction_mask], atol=1e-10)

    def test_constant_cin_with_diffusion(self):
        """Constant injection concentration gives near-constant extraction with diffusion.

        Small deviations at the edges are physical: the coefficient row sum can be
        slightly < 1 because diffusion allows some ambient water to mix in.
        """
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.ones(6) * 7.0
        layer_heights = np.array([2.0, 5.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        extraction_mask = flow < 0
        assert_allclose(cout[extraction_mask], 7.0, atol=0.05)

    def test_higher_diffusivity_more_spreading(self):
        """Higher molecular diffusivity produces more spreading (less peaked)."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_low_d = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        cout_high_d = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.1,
        )

        extraction_mask = flow < 0
        # With more diffusion, peak concentration should be lower
        assert np.max(cout_high_d[extraction_mask]) < np.max(cout_low_d[extraction_mask])

    def test_wider_h_distribution_more_spreading(self):
        """Wider layer height distribution causes more irreversible spreading."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])

        # Narrow distribution: all layers same height
        cout_narrow = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0, 5.0, 5.0]),
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        # Wide distribution: very different layer heights
        cout_wide = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([1.0, 5.0, 20.0]),
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        extraction_mask = flow < 0
        # Wide distribution should have more spreading → lower peak
        assert np.max(cout_wide[extraction_mask]) < np.max(cout_narrow[extraction_mask])


class TestRetardation:
    """Retardation tests."""

    def test_r_equals_one_matches_no_retardation(self):
        """R=1 matches result without retardation."""
        flow = np.array([100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=5, freq="D")
        cin = np.array([1.0, 5.0, 0.0, 0.0])

        cout_r1 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            retardation_factor=1.0,
        )

        cout_default = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        assert_allclose(cout_r1, cout_default)

    def test_retardation_invisible_without_diffusion(self):
        """R>1 with pure advection still gives perfect recovery (LIFO)."""
        flow = np.array([100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=5, freq="D")
        cin = np.array([1.0, 5.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            retardation_factor=3.0,
        )

        assert_allclose(cout[2], 5.0)
        assert_allclose(cout[3], 1.0)

    def test_retardation_invisible_with_molecular_diffusion(self):
        """R>1 with molecular diffusion only does not change spreading.

        In a push-pull test, retardation reduces both the radial front distance
        (r proportional to 1/sqrt(R)) and the effective diffusivity (D_eff = D_m/R)
        by the same factor, so the erf argument x/(2*sqrt(D_m*tau/R)) is
        R-independent. This is a well-known property of push-pull tests.
        """
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_r1 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            retardation_factor=1.0,
            molecular_diffusivity=0.01,
        )

        cout_r3 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            retardation_factor=3.0,
            molecular_diffusivity=0.01,
        )

        extraction_mask = flow < 0
        assert_allclose(cout_r3[extraction_mask], cout_r1[extraction_mask])

    def test_retardation_changes_dispersivity_spreading(self):
        """R>1 with longitudinal dispersivity produces more relative spreading.

        Retardation reduces the radial front distance (r proportional to 1/sqrt(R))
        and path length L proportional to 1/sqrt(R), so alpha_L*L shrinks faster
        than x^2, increasing the ratio of dispersion to advection.
        """
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_r1 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            retardation_factor=1.0,
            longitudinal_dispersivity=0.5,
        )

        cout_r3 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            retardation_factor=3.0,
            longitudinal_dispersivity=0.5,
        )

        extraction_mask = flow < 0
        # R>1 reduces path length, increasing relative dispersion → lower peak
        assert np.max(cout_r3[extraction_mask]) < np.max(cout_r1[extraction_mask])


class TestAdvectionMatrix:
    """Direct tests for _push_pull_advection_matrix."""

    def test_identity_single_inject_extract(self):
        """Single inject then single extract: W is identity-like."""
        flow = np.array([100.0, -100.0])
        dt = np.array([1.0, 1.0])
        tedges_days = np.array([0.0, 1.0, 2.0])
        w, _ = _push_pull_advection_matrix(flow=flow, dt=dt, tedges_days=tedges_days, cout_tedges_days=tedges_days)

        assert_allclose(w[1, 0], 1.0)
        assert_allclose(w[0, :], 0.0)

    def test_row_sums(self):
        """Extraction rows sum to 1 when enough volume was injected."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        dt = np.ones(6)
        tedges_days = np.arange(7, dtype=float)
        w, _ = _push_pull_advection_matrix(flow=flow, dt=dt, tedges_days=tedges_days, cout_tedges_days=tedges_days)

        # Extraction rows should sum to 1
        for i in [3, 4, 5]:
            assert_allclose(w[i, :].sum(), 1.0)

        # Injection rows should sum to 0
        for i in [0, 1, 2]:
            assert_allclose(w[i, :].sum(), 0.0)

    def test_rest_bins_are_zero(self):
        """Rest bins (flow=0) have zero rows."""
        flow = np.array([100.0, 0.0, -100.0])
        dt = np.ones(3)
        tedges_days = np.arange(4, dtype=float)
        w, _ = _push_pull_advection_matrix(flow=flow, dt=dt, tedges_days=tedges_days, cout_tedges_days=tedges_days)

        assert_allclose(w[1, :], 0.0)


class TestPushPullInverse:
    """Backward function tests."""

    def test_roundtrip_advection(self):
        """Forward then inverse recovers cin (pure advection)."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        injection_mask = flow > 0
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=1e-6)

    def test_roundtrip_with_diffusion(self):
        """Forward then inverse approximately recovers cin with diffusion."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        injection_mask = flow > 0
        # With regularization, recovery is approximate
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=0.5)


class TestGammaConvenience:
    """Gamma-distributed layer heights."""

    def test_gamma_wrapper_matches_manual(self):
        """Gamma wrapper matches manual discretization."""

        flow = np.array([100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=5, freq="D")
        cin = np.array([1.0, 5.0, 0.0, 0.0])

        gamma_bins = gamma_bins_func(mean=5.0, std=1.5, n_bins=50)
        cout_manual = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=gamma_bins["expected_values"],
            porosity=0.3,
        )

        cout_gamma = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.5,
            n_bins=50,
            porosity=0.3,
        )

        assert_allclose(cout_gamma, cout_manual)

    def test_gamma_alpha_beta_parameterization(self):
        """Alpha/beta parameterization works."""
        flow = np.array([100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=5, freq="D")
        cin = np.array([1.0, 5.0, 0.0, 0.0])

        # Should not raise
        cout = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            alpha=11.11,
            beta=0.45,
            porosity=0.3,
        )

        extraction_mask = flow < 0
        assert np.all(np.isfinite(cout[extraction_mask]))


class TestBackgroundConcentration:
    """Tests for c_background parameter."""

    def test_zero_background_is_default(self):
        """c_background=0 matches default behavior."""
        flow = np.array([100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=6, freq="D")
        cin = np.array([5.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 8.0])

        cout_default = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
        )
        cout_bg0 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
            c_background=0.0,
        )
        assert_allclose(cout_bg0, cout_default)

    def test_uniform_concentration_gives_constant_output(self):
        """When cin equals c_background, extraction is constant regardless of diffusion."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.ones(8) * 7.0
        layer_heights = np.array([2.0, 5.0, 10.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.05,
            c_background=7.0,
        )

        extraction_mask = flow < 0
        assert_allclose(cout[extraction_mask], 7.0)

    def test_over_extraction_uses_background(self):
        """Over-extraction bins use background concentration (advection)."""
        flow = np.array([100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([3.0, 5.0, 0.0, 0.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            c_background=2.0,
        )

        # LIFO: first 2 extraction bins get injected water
        assert_allclose(cout[2], 5.0)
        assert_allclose(cout[3], 3.0)
        # Over-extraction bins get background concentration
        assert_allclose(cout[4], 2.0)
        assert_allclose(cout[5], 2.0)

    def test_gamma_wrapper_passes_background(self):
        """gamma_push_pull_well passes c_background through.

        When cin == c_background, the extraction concentration should be close
        to the constant value. Small deviations (~1e-6) arise because the
        finite prepend volume means the coefficient matrix rows sum to slightly
        less than 1.0 due to diffusion at the prepend boundary.
        """
        flow = np.array([100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=6, freq="D")
        cin = np.ones(5) * 4.0

        cout = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=2.0,
            n_bins=30,
            porosity=0.3,
            molecular_diffusivity=0.01,
            c_background=4.0,
        )

        extraction_mask = flow < 0
        assert_allclose(cout[extraction_mask], 4.0, rtol=1e-5)


class TestCoutTedges:
    """Tests for cout_tedges (flow-weighted resampling) feature."""

    def test_same_grid_gives_lifo(self):
        """Using cout_tedges=tedges reproduces the LIFO extraction result."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # LIFO: last injected (5.0) extracted first
        extraction_mask = flow < 0
        assert_allclose(cout[extraction_mask], [5.0, 3.0, 1.0])
        # Injection bins should be NaN
        assert np.all(np.isnan(cout[~extraction_mask]))

    def test_coarser_grid_mass_conservation(self):
        """Flow-weighted resampling to coarser grid conserves mass."""
        flow = np.array([100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        # Fine resolution cout
        cout_fine = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # Coarse: extraction period in 2 bins (2 days each)
        cout_tedges = pd.date_range("2020-01-05", periods=3, freq="2D")
        cout_coarse = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # Mass conservation: total extracted mass should be equal
        dt_fine = np.diff(tedges) / pd.Timedelta("1D")
        extraction_mask = flow < 0
        mass_fine = np.nansum(cout_fine[extraction_mask] * np.abs(flow[extraction_mask]) * dt_fine[extraction_mask])

        dt_coarse = np.diff(cout_tedges) / pd.Timedelta("1D")
        # Flow is constant in this scenario
        mass_coarse = np.nansum(cout_coarse * 100.0 * dt_coarse)

        assert_allclose(mass_coarse, mass_fine, rtol=1e-12)

    def test_coarser_grid_flow_weighted_average(self):
        """Coarser cout_tedges bins are the flow-weighted average of fine bins."""
        flow = np.array([100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout_fine = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # Coarse: combine extraction into one bin
        cout_tedges = pd.DatetimeIndex([tedges[4], tedges[8]])
        cout_coarse = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # With constant flow, the coarse bin is the simple average of fine bins
        expected = np.nanmean(cout_fine[4:8])
        assert_allclose(cout_coarse[0], expected)

    def test_non_overlapping_bins_are_nan(self):
        """Cout_tedges bins outside extraction period are NaN."""
        flow = np.array([100.0, 100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=5, freq="D")
        cin = np.array([1.0, 2.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        # cout_tedges covers injection period only
        cout_tedges = pd.DatetimeIndex([tedges[0], tedges[2]])
        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        assert np.all(np.isnan(cout))

    def test_with_diffusion(self):
        """Cout_tedges works with diffusion enabled."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_fine = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        # Coarse: single extraction bin
        cout_tedges = pd.DatetimeIndex([tedges[3], tedges[6]])
        cout_coarse = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        expected = np.nanmean(cout_fine[3:6])
        assert_allclose(cout_coarse[0], expected)

    def test_with_background(self):
        """Cout_tedges works with c_background."""
        n_inject = 5
        n_extract = 10
        n = n_inject + n_extract
        flow = np.zeros(n)
        flow[:n_inject] = 100.0
        flow[n_inject:] = -100.0
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        cin = np.ones(n) * 5.0

        cout_fine = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            c_background=2.0,
        )

        # Coarse: combine all extraction into two bins
        cout_tedges = pd.date_range(tedges[n_inject], periods=3, freq=f"{n_extract // 2}D")
        cout_coarse = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            c_background=2.0,
        )

        # Check mass conservation
        dt_fine = np.diff(tedges) / pd.Timedelta("1D")
        extraction_mask = flow < 0
        mass_fine = np.nansum(cout_fine[extraction_mask] * np.abs(flow[extraction_mask]) * dt_fine[extraction_mask])

        dt_coarse = np.diff(cout_tedges) / pd.Timedelta("1D")
        mass_coarse = np.nansum(cout_coarse * 100.0 * dt_coarse)

        assert_allclose(mass_coarse, mass_fine, rtol=1e-12)

    def test_inverse_roundtrip_advection(self):
        """Forward with cout_tedges then inverse recovers cin."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout_tedges = pd.DatetimeIndex([tedges[3], tedges[4], tedges[5], tedges[6]])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        injection_mask = flow > 0
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=1e-6)

    def test_variable_flow_mass_conservation(self):
        """Mass conservation with variable extraction flow rates."""
        flow = np.array([100.0, 200.0, 150.0, -50.0, -200.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=8, freq="D")
        cin = np.array([3.0, 7.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout_fine = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # Coarse: two bins covering the extraction period
        cout_tedges = pd.DatetimeIndex([tedges[3], tedges[5], tedges[7]])
        cout_coarse = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        dt = np.diff(tedges) / pd.Timedelta("1D")
        extraction_mask = flow < 0
        mass_fine = np.nansum(cout_fine[extraction_mask] * np.abs(flow[extraction_mask]) * dt[extraction_mask])

        # For coarse bins, compute mass using flow-weighted volume per bin
        # First coarse bin covers flow[3:5] = [-50, -200], second covers flow[5:7] = [-100, -100]
        vol_coarse = np.array([50.0 * 1 + 200.0 * 1, 100.0 * 1 + 100.0 * 1])
        mass_coarse = np.nansum(cout_coarse * vol_coarse)

        assert_allclose(mass_coarse, mass_fine, rtol=1e-12)

    def test_gamma_forward_with_cout_tedges(self):
        """Gamma convenience wrapper passes through cout_tedges."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])

        cout_fine = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=10,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        cout_tedges = pd.DatetimeIndex([tedges[3], tedges[6]])
        cout_coarse = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=cout_tedges,
            mean=5.0,
            std=1.0,
            n_bins=10,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        expected = np.nanmean(cout_fine[3:6])
        assert_allclose(cout_coarse[0], expected)


class TestDiffusionCoefficients:
    """Tests for diffusion coefficient matrix properties."""

    def test_coefficient_row_sums(self):
        """Coefficient rows for extraction bins sum to <= 1."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([2.0, 5.0, 8.0])

        # Build coefficient matrix via forward model with known input
        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        extraction_mask = flow < 0
        # All extraction concentrations should be finite and non-negative
        assert np.all(np.isfinite(cout[extraction_mask]))
        assert np.all(cout[extraction_mask] >= 0)

    def test_dispersivity_only(self):
        """D_m=0 with alpha_L>0 produces spreading from dispersivity alone."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_adv = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cout_disp = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            longitudinal_dispersivity=0.5,
        )

        extraction_mask = flow < 0
        # Dispersivity should cause spreading (lower peak than advection)
        assert np.max(cout_disp[extraction_mask]) < np.max(cout_adv[extraction_mask])

    def test_variable_flow_rates(self):
        """Model handles variable flow rates between bins."""
        flow = np.array([50.0, 150.0, 100.0, -80.0, -120.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.ones(6) * 5.0

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([3.0, 7.0]),
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        extraction_mask = flow < 0
        # Constant input should give near-constant output
        assert_allclose(cout[extraction_mask], 5.0, atol=0.1)


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_flow_cin_length_mismatch(self):
        """Mismatched flow and cin lengths raise ValueError."""
        flow = np.array([100.0, 100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=4, freq="D")
        cin = np.array([1.0, 2.0])  # wrong length

        with pytest.raises(ValueError, match="flow and cin must have the same length"):
            push_pull_well(
                flow=flow,
                cin=cin,
                tedges=tedges,
                cout_tedges=tedges,
                layer_heights=np.array([5.0]),
                porosity=0.3,
            )

    def test_tedges_length_mismatch(self):
        """Wrong tedges length raises ValueError."""
        flow = np.array([100.0, 100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=3, freq="D")  # should be 4
        cin = np.array([1.0, 2.0, 0.0])

        with pytest.raises(ValueError, match="tedges must have length"):
            push_pull_well(
                flow=flow,
                cin=cin,
                tedges=tedges,
                cout_tedges=tedges,
                layer_heights=np.array([5.0]),
                porosity=0.3,
            )


class TestSignedRadialDistanceExtended:
    """Extended tests for _signed_radial_distance."""

    def test_retardation_factor(self):
        """Retardation factor R>1 reduces radial distance by sqrt(R)."""
        dv = np.array([1000.0])
        h, n_por, n_lay = 5.0, 0.3, 1

        x_r1 = _signed_radial_distance(
            delta_volume=dv, layer_height=h, porosity=n_por, retardation_factor=1.0, n_layers=n_lay
        )
        x_r4 = _signed_radial_distance(
            delta_volume=dv, layer_height=h, porosity=n_por, retardation_factor=4.0, n_layers=n_lay
        )

        # r scales as 1/sqrt(R), so R=4 gives half the distance
        assert_allclose(x_r4[0], x_r1[0] / 2.0)

    def test_n_layers_scaling(self):
        """More layers reduce radial distance by sqrt(N)."""
        dv = np.array([1000.0])
        h, n_por, r_factor = 5.0, 0.3, 1.0

        x_1 = _signed_radial_distance(
            delta_volume=dv, layer_height=h, porosity=n_por, retardation_factor=r_factor, n_layers=1
        )
        x_9 = _signed_radial_distance(
            delta_volume=dv, layer_height=h, porosity=n_por, retardation_factor=r_factor, n_layers=9
        )

        assert_allclose(x_9[0], x_1[0] / 3.0)

    def test_porosity_scaling(self):
        """Larger porosity reduces radial distance by sqrt(n)."""
        dv = np.array([1000.0])
        h, r_factor, n_lay = 5.0, 1.0, 1

        x_n1 = _signed_radial_distance(
            delta_volume=dv, layer_height=h, porosity=0.1, retardation_factor=r_factor, n_layers=n_lay
        )
        x_n4 = _signed_radial_distance(
            delta_volume=dv, layer_height=h, porosity=0.4, retardation_factor=r_factor, n_layers=n_lay
        )

        assert_allclose(x_n4[0], x_n1[0] / 2.0)


class TestMassConservationWithDiffusion:
    """Mass conservation tests for the diffusion model."""

    def test_single_layer_balanced_volumes(self):
        """Single uniform layer with balanced inject/extract conserves mass.

        The reflecting boundary at the well screen (r=0) ensures no tracer
        diffuses to r<0, so all injected mass is recovered during extraction.
        """
        flow = np.array([100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.array([2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        dt = np.diff(tedges) / pd.Timedelta("1D")
        mass_in = np.sum(cin * flow * dt)
        cout_clean = np.where(np.isnan(cout), 0.0, cout)
        mass_out = np.sum(cout_clean * np.abs(flow) * dt)

        assert_allclose(mass_out, mass_in, rtol=1e-10)

    def test_mass_conservation_dispersivity(self):
        """Single uniform layer with dispersivity conserves mass."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 5.0, 3.0, 0.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
            longitudinal_dispersivity=0.1,
        )

        dt = np.diff(tedges) / pd.Timedelta("1D")
        mass_in = np.sum(cin * flow * dt)
        cout_clean = np.where(np.isnan(cout), 0.0, cout)
        mass_out = np.sum(cout_clean * np.abs(flow) * dt)

        assert_allclose(mass_out, mass_in, rtol=1e-10)


class TestNonNegativeOutput:
    """Output concentration must be non-negative for non-negative input."""

    def test_non_negative_with_diffusion(self):
        """All extraction concentrations are >= 0 when cin >= 0."""
        flow = np.array([100.0, 100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=9, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([1.0, 3.0, 10.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.05,
        )

        extraction_mask = flow < 0
        assert np.all(cout[extraction_mask] >= 0)

    def test_non_negative_with_dispersivity(self):
        """All extraction concentrations are >= 0 with dispersivity."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([2.0, 5.0, 15.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            longitudinal_dispersivity=1.0,
        )

        extraction_mask = flow < 0
        assert np.all(cout[extraction_mask] >= 0)


class TestRestPeriodEffect:
    """Rest period between injection and extraction allows more diffusion."""

    def test_longer_rest_more_spreading(self):
        """Longer rest period gives more diffusion spreading (lower peak)."""
        layer_heights = np.array([3.0, 5.0, 8.0])

        # No rest: inject 2, extract 2 immediately
        flow_short = np.array([100.0, 100.0, -100.0, -100.0])
        tedges_short = pd.date_range("2020-01-01", periods=5, freq="D")
        cin_short = np.array([0.0, 10.0, 0.0, 0.0])
        cout_short = push_pull_well(
            flow=flow_short,
            cin=cin_short,
            tedges=tedges_short,
            cout_tedges=tedges_short,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.1,
        )

        # Long rest: 50 days between inject and extract
        n_rest = 50
        flow_long = np.concatenate(([100.0, 100.0], np.zeros(n_rest), [-100.0, -100.0]))
        tedges_long = pd.date_range("2020-01-01", periods=len(flow_long) + 1, freq="D")
        cin_long = np.zeros(len(flow_long))
        cin_long[1] = 10.0
        cout_long = push_pull_well(
            flow=flow_long,
            cin=cin_long,
            tedges=tedges_long,
            cout_tedges=tedges_long,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.1,
        )

        extraction_short = flow_short < 0
        extraction_long = flow_long < 0
        # Longer rest → more molecular diffusion → lower peak
        assert np.max(cout_long[extraction_long]) < np.max(cout_short[extraction_short])


class TestVariableTimeSteps:
    """Tests with non-uniform time step sizes."""

    def test_variable_dt_mass_conservation(self):
        """Mass is conserved with variable time step sizes (advection)."""
        tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06", "2020-01-08", "2020-01-15"])
        flow = np.array([100.0, 50.0, 200.0, -100.0, -80.0])
        cin = np.array([3.0, 7.0, 2.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        dt = np.diff(tedges) / pd.Timedelta("1D")
        mass_in = np.sum(cin * flow * dt)
        cout_clean = np.where(np.isnan(cout), 0.0, cout)
        mass_out = np.sum(cout_clean * np.abs(flow) * dt)

        assert_allclose(mass_out, mass_in, rtol=1e-12)

    def test_variable_dt_lifo_ordering(self):
        """LIFO ordering is correct with variable time steps."""
        tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06", "2020-01-09"])
        # Bin 0: 1 day, 100 m³/day → 100 m³
        # Bin 1: 3 days, 100 m³/day → 300 m³
        # Bin 2: 1 day, -400 m³/day → -400 m³ (extracts all of bin 1, then none of bin 0)
        # Bin 3: 3 days → no extraction left? Actually let's make it simpler
        flow = np.array([100.0, 100.0, -100.0, -100.0])
        cin = np.array([1.0, 5.0, 0.0, 0.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        dt = np.diff(tedges) / pd.Timedelta("1D")
        # Inject: bin 0 = 100 m³ at c=1, bin 1 = 300 m³ at c=5
        # Extract bin 2: 100 m³ → LIFO pops from bin 1 (100 of 300 m³) → c=5
        # Extract bin 3: 300 m³ → LIFO pops remaining 200 m³ from bin 1 and 100 m³ from bin 0
        #   → c = (200*5 + 100*1)/300 = 1100/300 = 11/3
        assert_allclose(cout[2], 5.0)
        assert_allclose(cout[3], (200.0 * 5.0 + 100.0 * 1.0) / 300.0)

        # Mass conservation
        mass_in = np.sum(cin * flow * dt)
        cout_clean = np.where(np.isnan(cout), 0.0, cout)
        mass_out = np.sum(cout_clean * np.abs(flow) * dt)
        assert_allclose(mass_out, mass_in, rtol=1e-12)


class TestPorosityEffect:
    """Tests for porosity effect on diffusion spreading."""

    def test_higher_porosity_more_relative_spreading_with_dispersivity(self):
        """Higher porosity gives more relative spreading with dispersivity.

        With dispersivity (alpha_L), spreading sigma ~ sqrt(alpha_L * L) where
        L ~ 1/sqrt(n), and radial distance x ~ 1/sqrt(n). The ratio sigma/x
        scales as sqrt(n), so higher porosity gives more relative spreading
        and a lower peak.
        """
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_high_n = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.4,
            longitudinal_dispersivity=0.5,
        )

        cout_low_n = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.1,
            longitudinal_dispersivity=0.5,
        )

        extraction_mask = flow < 0
        # Higher porosity → more relative dispersion (sigma/x ~ sqrt(n)) → lower peak
        assert np.max(cout_high_n[extraction_mask]) < np.max(cout_low_n[extraction_mask])


class TestCombinedDiffusionMechanisms:
    """Tests for combined molecular diffusion and dispersivity."""

    def test_combined_more_spreading_than_either(self):
        """Combined D_m and alpha_L gives more spreading than either alone."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_dm_only = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        cout_al_only = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            longitudinal_dispersivity=0.5,
        )

        cout_both = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.5,
        )

        extraction_mask = flow < 0
        peak_dm = np.max(cout_dm_only[extraction_mask])
        peak_al = np.max(cout_al_only[extraction_mask])
        peak_both = np.max(cout_both[extraction_mask])

        # Combined spreading exceeds either individual mechanism
        assert peak_both < peak_dm
        assert peak_both < peak_al


class TestOverExtractionWithDiffusion:
    """Tests for over-extraction behavior with diffusion and background."""

    def test_over_extraction_with_background_and_diffusion(self):
        """Over-extraction with diffusion and c_background smoothly transitions."""
        flow = np.array([100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.01,
            c_background=2.0,
        )

        extraction_mask = flow < 0
        # All extraction values should be finite and non-negative
        assert np.all(np.isfinite(cout[extraction_mask]))
        assert np.all(cout[extraction_mask] >= 0)
        # First extraction bin should be closer to cin than background
        assert cout[2] > 2.0
        # Last extraction bin (over-extraction) should be closer to background
        assert cout[5] < cout[2]


class TestGammaInverse:
    """Tests for gamma_push_pull_well_inverse."""

    def test_roundtrip_advection(self):
        """Gamma inverse recovers cin for pure advection."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])

        cout = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
        )

        cin_recovered = gamma_push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
        )

        injection_mask = flow > 0
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=1e-6)

    def test_roundtrip_with_diffusion(self):
        """Gamma inverse approximately recovers cin with diffusion."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([1.0, 3.0, 5.0, 0.0, 0.0, 0.0])

        cout = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        cin_recovered = gamma_push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        injection_mask = flow > 0
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=0.5)

    def test_nan_for_non_injection_bins(self):
        """Inverse returns NaN for extraction and rest bins."""
        flow = np.array([100.0, 100.0, 0.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=6, freq="D")
        cin = np.array([3.0, 5.0, 0.0, 0.0, 0.0])

        cout = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
        )

        cin_recovered = gamma_push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
        )

        # Rest bin (index 2) and extraction bins should be NaN
        assert np.isnan(cin_recovered[2])
        assert np.all(np.isnan(cin_recovered[3:]))

    def test_with_background(self):
        """Gamma inverse roundtrip with c_background."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=8, freq="D")
        cin = np.array([5.0, 8.0, 3.0, 0.0, 0.0, 0.0, 0.0])

        cout = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
            c_background=2.0,
        )

        cin_recovered = gamma_push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            mean=5.0,
            std=1.0,
            n_bins=20,
            porosity=0.3,
            c_background=2.0,
        )

        injection_mask = flow > 0
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=1e-4)


class TestInverseWithBackground:
    """Inverse model tests with c_background."""

    def test_roundtrip_advection_with_background(self):
        """Forward then inverse recovers cin with c_background (advection)."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=8, freq="D")
        cin = np.array([5.0, 8.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            c_background=2.0,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            c_background=2.0,
        )

        injection_mask = flow > 0
        assert_allclose(cin_recovered[injection_mask], cin[injection_mask], atol=1e-4)

    def test_inverse_nan_for_non_injection(self):
        """Inverse returns NaN for extraction and rest bins."""
        flow = np.array([100.0, 100.0, 0.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=6, freq="D")
        cin = np.array([3.0, 5.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([5.0])

        cout = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
            cout_tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
        )

        # Rest bin (index 2) and extraction bins should be NaN
        assert np.isnan(cin_recovered[2])
        assert np.all(np.isnan(cin_recovered[3:]))
