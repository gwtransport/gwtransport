"""Tests for radial push-pull well transport model."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from gwtransport.gamma import bins as gamma_bins_func
from gwtransport.radial import (
    _push_pull_advection_matrix,
    _signed_radial_distance,
    gamma_push_pull_well,
    push_pull_well,
    push_pull_well_inverse,
)


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
            layer_heights=np.array([5.0]),
            porosity=0.3,
        )

        # Multiple heterogeneous layers
        cout_hetero = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
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
            well_radius=0.0,
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
            well_radius=0.0,
            n_layers=1,
        )
        assert x[0] > 0
        assert x[1] < 0
        assert_allclose(np.abs(x[0]), np.abs(x[1]))

    def test_well_radius_reduces_distance(self):
        """Nonzero well radius reduces radial distance from well."""
        dv = np.array([1000.0])
        x_no_well = _signed_radial_distance(
            delta_volume=dv,
            layer_height=5.0,
            porosity=0.3,
            retardation_factor=1.0,
            well_radius=0.0,
            n_layers=1,
        )
        x_with_well = _signed_radial_distance(
            delta_volume=dv,
            layer_height=5.0,
            porosity=0.3,
            retardation_factor=1.0,
            well_radius=0.5,
            n_layers=1,
        )
        assert x_with_well[0] < x_no_well[0]

    def test_formula(self):
        """Verify against manual calculation."""
        dv = np.array([1000.0])
        h, n_por, r_factor, rw, n_lay = 5.0, 0.3, 1.0, 0.1, 3
        scale = n_lay * np.pi * h * n_por * r_factor
        expected = np.sqrt(rw**2 + 1000.0 / scale) - rw

        x = _signed_radial_distance(
            delta_volume=dv,
            layer_height=h,
            porosity=n_por,
            retardation_factor=r_factor,
            well_radius=rw,
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
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cout_diff = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
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
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        cout_high_d = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
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
            layer_heights=np.array([5.0, 5.0, 5.0]),
            porosity=0.3,
            molecular_diffusivity=0.01,
        )

        # Wide distribution: very different layer heights
        cout_wide = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
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
            layer_heights=np.array([5.0]),
            porosity=0.3,
            retardation_factor=1.0,
        )

        cout_default = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
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
            layer_heights=np.array([5.0]),
            porosity=0.3,
            retardation_factor=3.0,
        )

        assert_allclose(cout[2], 5.0)
        assert_allclose(cout[3], 1.0)

    def test_retardation_changes_diffusion_spreading(self):
        """R>1 with molecular diffusion produces more relative spreading.

        Retardation reduces the radial front distance (r proportional to 1/sqrt(R))
        but molecular diffusion magnitude is unchanged. This increases the
        ratio of diffusion to advection, causing more spreading and a lower peak.
        """
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        tedges = pd.date_range("2020-01-01", periods=7, freq="D")
        cin = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        layer_heights = np.array([3.0, 5.0, 8.0])

        cout_r1 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            retardation_factor=1.0,
            molecular_diffusivity=0.01,
        )

        cout_r3 = push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
            layer_heights=layer_heights,
            porosity=0.3,
            retardation_factor=3.0,
            molecular_diffusivity=0.01,
        )

        extraction_mask = flow < 0
        # R>1 reduces radial distance but D_m*tau unchanged → more relative spreading
        assert np.max(cout_r3[extraction_mask]) < np.max(cout_r1[extraction_mask])


class TestAdvectionMatrix:
    """Direct tests for _push_pull_advection_matrix."""

    def test_identity_single_inject_extract(self):
        """Single inject then single extract: W is identity-like."""
        flow = np.array([100.0, -100.0])
        dt = np.array([1.0, 1.0])
        w = _push_pull_advection_matrix(flow=flow, dt=dt)

        assert_allclose(w[1, 0], 1.0)
        assert_allclose(w[0, :], 0.0)

    def test_row_sums(self):
        """Extraction rows sum to 1 when enough volume was injected."""
        flow = np.array([100.0, 100.0, 100.0, -100.0, -100.0, -100.0])
        dt = np.ones(6)
        w = _push_pull_advection_matrix(flow=flow, dt=dt)

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
        w = _push_pull_advection_matrix(flow=flow, dt=dt)

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
            layer_heights=layer_heights,
            porosity=0.3,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
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
            layer_heights=layer_heights,
            porosity=0.3,
            molecular_diffusivity=0.001,
        )

        cin_recovered = push_pull_well_inverse(
            flow=flow,
            cout=cout,
            tedges=tedges,
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
            layer_heights=gamma_bins["expected_values"],
            porosity=0.3,
        )

        cout_gamma = gamma_push_pull_well(
            flow=flow,
            cin=cin,
            tedges=tedges,
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
            alpha=11.11,
            beta=0.45,
            porosity=0.3,
        )

        extraction_mask = flow < 0
        assert np.all(np.isfinite(cout[extraction_mask]))
