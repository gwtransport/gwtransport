"""Integration tests for ``gwtransport.advection.infiltration_to_extraction_nonlinear_sorption``."""

import warnings

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_nonlinear_sorption
from gwtransport.fronttracking.math import FreundlichSorption, LangmuirSorption, compute_first_front_arrival_theta
from gwtransport.fronttracking.solver import FrontTracker, find_unresolved_interaction
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.utils import compute_time_edges


class TestFrontTrackingAPI:
    """Integration tests for the public front-tracking API."""

    def test_basic_freundlich_sorption(self):
        """Test basic call with Freundlich sorption parameters."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))

    def test_constant_retardation(self):
        """Test with constant retardation factor (linear sorption)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            retardation_factor=2.0,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)

    def test_zero_concentration_input(self):
        """All-zero concentration input should yield zero output."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.zeros(len(dates))
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            retardation_factor=2.0,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert np.allclose(cout, 0.0)

    def test_parameter_validation(self):
        """Error if neither retardation_factor nor Freundlich params provided."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        with pytest.raises(ValueError, match="Must provide one of"):
            infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([500.0]),
            )

    def test_detailed_returns_consistent_structure(self):
        """Detailed API returns structure with consistent counts and sorption type."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, structures = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))

        # Check that we get one structure per pore volume
        assert len(structures) == 1
        structure = structures[0]

        waves = structure["waves"]
        n_shocks = sum(isinstance(w, ShockWave) for w in waves)
        n_rarefactions = sum(isinstance(w, RarefactionWave) for w in waves)
        n_characteristics = sum(isinstance(w, CharacteristicWave) for w in waves)

        assert structure["n_shocks"] == n_shocks
        assert structure["n_rarefactions"] == n_rarefactions
        assert structure["n_characteristics"] == n_characteristics

        assert isinstance(structure["sorption"], FreundlichSorption)

    def test_spinup_and_first_arrival_via_api(self):
        """θ-edge wiring + spin-up: cout is zero before arrival and plateaus at cin after.

        The window is sized to extend well past the first arrival (a small pore
        volume so breakthrough completes inside the window) and ``cin`` is
        sustained over the whole window so a steady plateau forms. This makes
        the "zero before arrival" check non-vacuous (it would otherwise compare
        zeros against zeros if the window ended before arrival) and adds the
        post-arrival plateau check. The θ-edge wiring is exercised end-to-end:
        ``theta_first_arrival`` must match the independent math helper, and the
        per-bin θ classification drives the spin-up boundary.
        """
        n_bins = 60
        dates = pd.date_range(start="2020-01-01", periods=n_bins, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        c_plateau = 5.0
        cin = np.array([0.0] + [c_plateau] * (n_bins - 1))
        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 100.0  # small so breakthrough completes inside the window

        freundlich_k = 0.01
        freundlich_n = 2.0
        bulk_density = 1500.0
        porosity = 0.3

        sorption = FreundlichSorption(
            k_f=freundlich_k,
            n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Build θ-edges and compute expected θ-first-arrival directly in θ-space.
        tedges_days = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
        theta_edges = np.concatenate(([0.0], np.cumsum(flow * np.diff(tedges_days))))
        theta_first_expected = compute_first_front_arrival_theta(
            cin=cin,
            theta_edges=theta_edges,
            aquifer_pore_volume=aquifer_pore_volume,
            sorption=sorption,
        )

        # Output bins coincide with the inlet window so the conservation form
        # stays inside the inlet θ-range (no post-inlet clamp/warning).
        cout_tedges = tedges

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cout, structures = infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([aquifer_pore_volume]),
                freundlich_k=freundlich_k,
                freundlich_n=freundlich_n,
                bulk_density=bulk_density,
                porosity=porosity,
            )

        assert len(structures) == 1
        structure = structures[0]
        assert structure["theta_first_arrival"] == pytest.approx(theta_first_expected, rel=1e-14)

        # Confirm the window genuinely extends past arrival (else the spin-up
        # check below would be vacuous).
        tracker_state = structure["tracker_state"]
        t_first = tracker_state.t_at_theta(theta_first_expected)
        cout_tedges_days = np.asarray((cout_tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
        assert cout_tedges_days[-1] > t_first

        # Spin-up: cout is exactly zero for bins whose upper-edge θ is before θ_first.
        # theta_at_t is monotone for positive flow, so theta_at_t(t_upper) < θ_first
        # iff t_upper < t_at_theta(θ_first) = t_first — vectorized directly.
        before_arrival = cout_tedges_days[1:] < t_first
        assert before_arrival.any(), "Expected at least one pre-arrival bin"
        assert np.all(cout[before_arrival] == 0.0)

        # Post-arrival: bins whose lower-edge θ is at/after θ_first plateau at cin.
        after_arrival = cout_tedges_days[:-1] >= t_first
        assert after_arrival.any(), "Expected at least one post-arrival bin"
        np.testing.assert_allclose(cout[after_arrival], c_plateau, atol=1e-13)

    def test_api_mass_conservation_nonlinear(self):
        """End-to-end mass conservation for a Freundlich n<1 pulse (API-C1).

        A finite pulse ``cin = [0]*5 + [10]*10 + [0]*…`` with a small pore
        volume lets the breakthrough complete inside the window. For ``n<1``
        the rarefaction tail is exact (no asymptotic residual), so the total
        outlet mass equals the injected mass to near machine precision. Mass is
        computed in θ-coordinates where ``flow·dt = Δθ`` per bin, so the t-bin
        sum ``Σ c·flow·dt`` is exactly ``∫ c dθ``. The whole call runs under
        ``warnings.simplefilter("error")`` to assert the degradation clamp
        never fires.

        Verified: rel_err is 0.0 on baseline and 1.0 against a ``return
        zeros`` mutation. ``n<1`` is required — for ``n>1`` the rarefaction
        tail is asymptotic (~0.08% residual even at 2000 d).
        """
        n_bins = 600
        cin = np.array([0.0] * 5 + [10.0] * 10 + [0.0] * (n_bins - 15))
        flow = np.full(n_bins, 100.0)
        aquifer_pore_volume = 50.0  # small so breakthrough completes within the window

        dates = pd.date_range(start="2020-01-01", periods=n_bins, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
        cout_tedges = tedges  # output bins coincide with inlet window

        freundlich_n = 0.5  # n<1: exact tail

        # Verify first arrival is INSIDE the window before relying on completion.
        sorption = FreundlichSorption(k_f=0.01, n=freundlich_n, bulk_density=1500.0, porosity=0.3)
        tedges_days = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
        theta_edges = np.concatenate(([0.0], np.cumsum(flow * np.diff(tedges_days))))
        theta_first = compute_first_front_arrival_theta(
            cin=cin, theta_edges=theta_edges, aquifer_pore_volume=aquifer_pore_volume, sorption=sorption
        )
        assert theta_first < theta_edges[-1], "First arrival must be inside the window"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cout, _ = infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([aquifer_pore_volume]),
                freundlich_k=0.01,
                freundlich_n=freundlich_n,
                bulk_density=1500.0,
                porosity=0.3,
            )

        # In θ-coordinates, flow·dt = Δθ per bin, so Σ c·flow·dt = ∫ c dθ.
        dtheta = flow * np.diff(tedges_days)
        mass_in = float(np.sum(cin * dtheta))
        mass_out = float(np.sum(cout * dtheta))
        np.testing.assert_allclose(mass_out, mass_in, rtol=1e-9)

    def test_api_constant_in_constant_out_nonlinear(self):
        """Constant inlet => constant outlet plateau after arrival, for varying flow.

        With ``cin ≡ 7`` everywhere, the post-arrival outlet must equal 7
        exactly regardless of the (time-varying) flow: a constant inlet has no
        wave structure beyond the leading front, so the steady-state outlet is
        the inlet value. The plateau assertion is unconditional (not gated on
        ``cout > 0``) so a ``return zeros`` mutation cannot slip through.
        """
        n_bins = 40
        c_const = 7.0
        cin = np.full(n_bins, c_const)
        # Time-varying flow exercises the V/θ→t mapping and flow-weighting.
        flow = np.linspace(80.0, 140.0, n_bins)
        aquifer_pore_volume = 50.0

        dates = pd.date_range(start="2020-01-01", periods=n_bins, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
        cout_tedges = tedges

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cout, structures = infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([aquifer_pore_volume]),
                freundlich_k=0.01,
                freundlich_n=2.0,
                bulk_density=1500.0,
                porosity=0.3,
            )

        tracker_state = structures[0]["tracker_state"]
        t_first = tracker_state.t_at_theta(structures[0]["theta_first_arrival"])
        cout_tedges_days = np.asarray((cout_tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)

        post_arrival = cout_tedges_days[:-1] >= t_first
        assert post_arrival.any(), "Expected post-arrival bins"
        np.testing.assert_allclose(cout[post_arrival], c_const, atol=1e-13)

    def test_api_freundlich_reduces_to_constant_r(self):
        """Freundlich n→1 reproduces the matching constant-retardation result.

        At ``n = 1`` Freundlich retardation is concentration-independent,
        ``R = 1 + ρ_b·k_f/θ_por``. For ``n = 1 ± δ`` the two outlet curves must
        converge to the constant-``R`` curve. The dominant discrepancy is the
        front-arrival timing shift: a δ-change in ``n`` shifts ``R`` by
        ``|dR/dn|·δ``, hence the arrival θ by ``V·|dR/dn|·δ``, changing the
        filled fraction of the single straddling bin by
        ``V·|dR/dn|·δ / Δθ_bin``. The resulting front-bin error is bounded by
        ``c_plateau · V·|dR/dn|·δ / Δθ_bin`` — an explicit O(δ) linearization
        bound derived from the isotherm, not a fudge factor.
        """
        n_bins = 40
        k_f = 0.01
        bulk_density = 1500.0
        porosity = 0.3
        aquifer_pore_volume = 50.0
        c_plateau = 5.0

        cin = np.array([0.0] * 3 + [c_plateau] * (n_bins - 3))
        flow_rate = 100.0
        flow = np.full(n_bins, flow_rate)

        dates = pd.date_range(start="2020-01-01", periods=n_bins, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
        cout_tedges = tedges

        # Constant R equal to the Freundlich n=1 limit.
        r_at_n1 = 1.0 + (bulk_density * k_f) / (porosity * 1.0)

        def run(*, freundlich_n=None, retardation_factor=None):
            kwargs = (
                {
                    "freundlich_k": k_f,
                    "freundlich_n": freundlich_n,
                    "bulk_density": bulk_density,
                    "porosity": porosity,
                }
                if freundlich_n is not None
                else {"retardation_factor": retardation_factor}
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                cout, _ = infiltration_to_extraction_nonlinear_sorption(
                    cin=cin,
                    flow=flow,
                    tedges=tedges,
                    cout_tedges=cout_tedges,
                    aquifer_pore_volumes=np.array([aquifer_pore_volume]),
                    **kwargs,
                )
            return cout

        delta = 1e-3
        cout_ref = run(retardation_factor=r_at_n1)

        # |dR/dn| at n=1 from the Freundlich isotherm (central difference).
        ref_sorption = FreundlichSorption(k_f=k_f, n=1.0 + 1e-6, bulk_density=bulk_density, porosity=porosity)
        sorption_lo = FreundlichSorption(k_f=k_f, n=1.0 - 1e-6, bulk_density=bulk_density, porosity=porosity)
        dr_dn = abs(float(ref_sorption.retardation(c_plateau)) - float(sorption_lo.retardation(c_plateau))) / 2e-6

        dtheta_bin = flow_rate * 1.0  # 1-day bins
        atol = c_plateau * aquifer_pore_volume * dr_dn / dtheta_bin * delta

        for freundlich_n in (1.0 - delta, 1.0 + delta):
            cout_fr = run(freundlich_n=freundlich_n)
            np.testing.assert_allclose(cout_fr, cout_ref, atol=atol)

    def test_api_freundlich_n_gt_1_n_greater_than_one(self):
        """API works for Freundlich with n>1 sorption (n>1)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,  # n>1 (n>1)
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))

    def test_api_freundlich_n_lt_1_n_less_than_one(self):
        """API works for Freundlich with n<1 sorption (n<1)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=0.5,  # n<1 (n<1)
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))

    def test_api_langmuir_sorption(self):
        """API works with Langmuir sorption parameters."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=20, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            langmuir_s_max=1.0,
            langmuir_k_l=5.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))

    def test_api_langmuir_detailed_returns_langmuir_sorption(self):
        """Detailed API returns LangmuirSorption in structure."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=20, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structures = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            langmuir_s_max=1.0,
            langmuir_k_l=5.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert len(structures) == 1
        assert isinstance(structures[0]["sorption"], LangmuirSorption)
        assert not np.any(np.isnan(cout))

    def test_api_conflicting_sorption_models_raises(self):
        """Error when both Freundlich and Langmuir parameters are provided."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        with pytest.raises(ValueError, match="Only one sorption model"):
            infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([500.0]),
                freundlich_k=0.01,
                freundlich_n=2.0,
                langmuir_s_max=1.0,
                langmuir_k_l=5.0,
                bulk_density=1500.0,
                porosity=0.3,
            )

    def test_api_langmuir_partial_params_raises(self):
        """Error when only some Langmuir parameters are provided."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        with pytest.raises(ValueError, match="All Langmuir parameters required"):
            infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([500.0]),
                langmuir_s_max=1.0,
                langmuir_k_l=5.0,
                # missing bulk_density and porosity
            )


class TestMultiPeakInteractionRefused:
    """End-to-end: a multi-peak cin whose fans overlap is refused, not silently mis-answered.

    A later pulse's front sweeps into the first pulse's decaying-shock fan — an unresolved wave
    interaction the front tracker does not compose. The single-owner reader then either drops the
    earlier pulse's near-inlet mass (leaking a spurious inlet→outlet echo in the conservation-form
    ``cout``) or over-counts it; neither is the true field. Rather than return that mass-losing
    output, the public API refuses the input with ``NotImplementedError`` (an output-side mitigation
    was tried and reverted because it fixed neither the physics nor a percolation regression). Exact
    multi-front interaction is deferred to a solver-level follow-up.
    """

    def test_multipeak_raises_not_implemented(self):
        """Overlapping-fan multi-peak input raises ``NotImplementedError`` at the public boundary."""
        cin = np.array([0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0])
        flow = np.full(len(cin), 100.0)
        tedges = pd.date_range("2020-01-01", periods=len(cin) + 1, freq="D")
        cout_tedges = pd.date_range("2020-01-01", periods=len(cin) + 1, freq="D")

        with pytest.raises(NotImplementedError, match="interacting fronts"):
            infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([40.0]),
                freundlich_k=0.05,
                freundlich_n=2.0,
                bulk_density=1600.0,
                porosity=0.35,
            )


def _fv_outlet_cout_freundlich_n2(cin, flow, v_pv, k_f, bulk_density, porosity, *, n_cells=2500, cfl=0.4):
    """Independent first-order upwind (Godunov) outlet oracle for Freundlich ``n = 2``.

    Marches the conserved variable ``U = C_T(c) = c + A·√c`` (``A = (ρ_b/φ)·k_f``) on a uniform
    ``V``-grid in cumulative flow θ, upwind flux ``F = c`` (all characteristic speeds ``dF/dU = 1/R``
    are in ``(0, 1]``, so upwind == Godunov and is entropy-correct for this convex flux). Returns the
    flow-weighted bin-averaged outlet concentration on the daily grid the public API uses, so the two
    are compared apples-to-apples over the same truncated window. Memory-safe: only the current cell
    profile plus scalar accumulators are stored.

    Parameters
    ----------
    cin : ndarray
        Inlet concentration per daily bin.
    flow : float
        Constant flow rate ``Q`` [m³/day].
    v_pv : float
        Outlet pore volume [m³].
    k_f, bulk_density, porosity : float
        Freundlich ``k_f`` and the ``ρ_b``, ``φ`` used to form ``A``.
    n_cells, cfl : int, float
        FV grid resolution and CFL number.

    Returns
    -------
    ndarray
        Daily bin-averaged outlet concentration, length ``len(cin)``.
    """
    a = (bulk_density / porosity) * k_f
    q = float(flow)
    cin = np.asarray(cin, dtype=float)
    nb = len(cin)
    theta_in_edges = q * np.arange(nb + 1)
    theta_out_edges = q * np.arange(nb + 1)
    dv = v_pv / n_cells
    cmax = max(cin.max(), 1e-12)
    r_min = 1.0 + a / (2.0 * np.sqrt(cmax))  # min retardation at cmax → max characteristic speed 1/r_min
    dtheta = cfl * dv * r_min

    def c_from_u(u):
        u = np.maximum(u, 0.0)
        x = (-a + np.sqrt(a * a + 4.0 * u)) / 2.0  # √c from x² + A x − U = 0
        return x * x

    c = np.zeros(n_cells)
    u = c + a * np.sqrt(c)
    theta = 0.0
    cum_out = 0.0
    out_edge_mass = np.zeros(len(theta_out_edges))
    next_edge = 0
    lam = dtheta / dv
    theta_max = theta_out_edges[-1]
    while theta < theta_max - 1e-12:
        while next_edge < len(theta_out_edges) and theta_out_edges[next_edge] <= theta + 1e-12:
            out_edge_mass[next_edge] = cum_out
            next_edge += 1
        if next_edge >= len(theta_out_edges):
            break
        cin_now = cin[min(int(theta // q), nb - 1)] if theta < theta_in_edges[-1] else cin[-1]
        c_left = np.empty(n_cells)
        c_left[0] = cin_now
        c_left[1:] = c[:-1]
        u -= lam * (c - c_left)
        c = c_from_u(u)
        cum_out += c[-1] * dtheta
        theta += dtheta
    while next_edge < len(theta_out_edges):
        out_edge_mass[next_edge] = cum_out
        next_edge += 1
    return np.diff(out_edge_mass) / np.diff(theta_out_edges)


class TestUnresolvedMultiFrontInteractionRaises:
    """The public API refuses inputs whose fronts interact unresolved, and runs non-interacting ones.

    The front tracker resolves shock↔shock / shock↔rarefaction collisions (the latter into a
    ``DecayingShockWave``) but never composes a later front overtaking an existing decaying-shock fan
    (nor two fans overlapping). Such an unresolved interaction makes the public ``cout`` a linear
    superposition of a nonlinear operator — mass-losing and silently wrong — so
    :func:`~gwtransport.advection.infiltration_to_extraction_nonlinear_sorption` raises
    ``NotImplementedError`` (via
    :func:`gwtransport.fronttracking.solver.find_unresolved_interaction`) rather than returning it.
    A single front, or well-separated pulses that break through before overtaking one another, run
    and match an independent Godunov FV oracle. Acceptance params: Freundlich ``n=2``, ``k_f=0.01``,
    ``ρ_b=1500``, ``φ=0.3``, ``flow=100``.
    """

    K_F, N, RHO_B, POR, FLOW = 0.01, 2.0, 1500.0, 0.3, 100.0

    def _run(self, pulse, v_pv, *, n_trail):
        cin = np.array(list(pulse) + [0.0] * n_trail, dtype=float)
        nb = len(cin)
        tedges = pd.date_range("2020-01-01", periods=nb + 1, freq="D")
        return infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=np.full(nb, self.FLOW),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([v_pv]),
            freundlich_k=self.K_F,
            freundlich_n=self.N,
            bulk_density=self.RHO_B,
            porosity=self.POR,
        )

    @pytest.mark.parametrize(
        ("pulse", "v_pv", "n_trail"),
        [
            # Geometric fan-overlap class (two fans share an in-domain point):
            ([0, 10, 10, 0, 0, 0, 5, 5, 0], 40.0, 111),  # multi-peak: pulse 2 sweeps pulse 1's decayed tail
            ([0, 5, 10, 10, 5, 0], 40.0, 74),  # plateau: internal c=10 shock catches the 0→5 shock before outlet
            ([0, 10, 10, 2, 2, 2, 5, 5, 0], 40.0, 111),  # nonzero-dip: a fan overtakes an earlier decaying fan
            # Conservation-symptom class (pulse-2 shock overtakes pulse-1's decaying-shock fan; the
            # two fans only cross BEYOND v_outlet, so the geometric scan cannot see it — the
            # cumulative-outlet-mass monotonicity check refuses the fabricated mass instead). These
            # are the seasonal/oscillating two-pulse signals a prior guard silently mis-answered.
            ([0, 10, 10, 0, 10, 10, 0], 40.0, 53),  # 2-pulse equal
            ([0, 5, 5, 0, 10, 10, 0], 40.0, 53),  # 2-pulse rising
            ([0, 10, 10, 0, 5, 5, 0], 40.0, 53),  # 2-pulse falling
        ],
    )
    def test_interacting_multifront_raises(self, pulse, v_pv, n_trail):
        """An in-domain unresolved wave interaction raises ``NotImplementedError``."""
        with pytest.raises(NotImplementedError, match="interacting fronts"):
            self._run(pulse, v_pv, n_trail=n_trail)

    def test_conservation_symptom_flags_fans_crossing_beyond_outlet(self):
        """The two-pulse refusal is driven by the mass-monotonicity net, not the geometric scan.

        For ``cin=[0,10,10,0,10,10,0]`` at ``V=40`` pulse-2's shock overtakes pulse-1's
        decaying-shock fan, but the two fans only cross at ``V≈97 ≫ v_outlet``: no in-domain point
        is ever covered by two fans, so the geometric fan-overlap scan returns clean. Cumulative
        outlet mass nonetheless drops (the reader over-counts stored mass), which the monotonicity
        detector reports. This pins that the added net — not the pre-existing geometric scan — is
        what catches this dominant multi-pulse class.
        """
        cin = np.array([0, 10, 10, 0, 10, 10, 0] + [0.0] * 53, dtype=float)
        nb = len(cin)
        tedges = pd.date_range("2020-01-01", periods=nb + 1, freq="D")
        sorption = FreundlichSorption(k_f=self.K_F, n=self.N, bulk_density=self.RHO_B, porosity=self.POR)
        tracker = FrontTracker(
            cin=cin, flow=np.full(nb, self.FLOW), tedges=tedges, aquifer_pore_volume=40.0, sorption=sorption
        )
        tracker.run()
        reason = find_unresolved_interaction(tracker.state)
        assert reason is not None, "interacting two-pulse input was not flagged"
        assert "cumulative outlet mass" in reason, f"expected mass-monotonicity attribution, got: {reason}"

    def test_plateau_below_interaction_volume_runs_and_matches_fv(self):
        """The SAME plateau input at a shorter column (V=30) exits before interacting — runs, matches FV.

        The c=10 shock would overtake the 0→5 shock near V≈31.6, past a V=30 outlet, so every front
        crosses the outlet before any unresolved collision: the detector must NOT fire (no false
        positive), and the outlet mass tracks the independent FV oracle to well within 1%.
        """
        pulse, v_pv, n_trail = [0, 5, 10, 10, 5, 0], 30.0, 74
        cout, _ = self._run(pulse, v_pv, n_trail=n_trail)
        cin = np.array(pulse + [0.0] * n_trail, dtype=float)
        cout_fv = _fv_outlet_cout_freundlich_n2(cin, self.FLOW, v_pv, self.K_F, self.RHO_B, self.POR)
        m_pub = float(np.nansum(cout))
        m_fv = float(np.nansum(cout_fv))
        assert abs(m_pub - m_fv) < 0.01 * m_fv, f"plateau V30 outlet mass {m_pub:.3f} vs FV {m_fv:.3f}"

    @pytest.mark.parametrize("v_pv", [20.0, 30.0, 40.0])
    def test_single_pulse_runs_and_matches_fv(self, v_pv):
        """A single returning-to-zero pulse is a single front (one DSW) — runs and matches FV to <1%."""
        pulse, n_trail = [0, 10, 10, 0], 40
        cout, _ = self._run(pulse, v_pv, n_trail=n_trail)
        cin = np.array(pulse + [0.0] * n_trail, dtype=float)
        cout_fv = _fv_outlet_cout_freundlich_n2(cin, self.FLOW, v_pv, self.K_F, self.RHO_B, self.POR)
        m_pub = float(np.nansum(cout))
        m_fv = float(np.nansum(cout_fv))
        assert abs(m_pub - m_fv) < 0.01 * m_fv, f"single pulse V{v_pv} outlet mass {m_pub:.3f} vs FV {m_fv:.3f}"

    def test_monotone_staircase_runs_and_matches_fv(self):
        """A rising multi-step input (single rarefaction, no interaction) must NOT trip the net.

        ``cin=[0,3,6,9,12,0]`` climbs monotonically, forming one spreading rarefaction plus the
        trailing shock — no shock overtakes another fan, cumulative outlet mass stays monotone.
        Guards the conservation-symptom detector against false-firing on legitimate multi-structure
        (non-single-pulse) inputs; outlet mass tracks the FV oracle to well within 1%.
        """
        pulse, v_pv, n_trail = [0, 3, 6, 9, 12, 0], 40.0, 74
        cout, _ = self._run(pulse, v_pv, n_trail=n_trail)
        cin = np.array(pulse + [0.0] * n_trail, dtype=float)
        cout_fv = _fv_outlet_cout_freundlich_n2(cin, self.FLOW, v_pv, self.K_F, self.RHO_B, self.POR)
        m_pub = float(np.nansum(cout))
        m_fv = float(np.nansum(cout_fv))
        assert abs(m_pub - m_fv) < 0.01 * m_fv, f"staircase V{v_pv} outlet mass {m_pub:.3f} vs FV {m_fv:.3f}"
