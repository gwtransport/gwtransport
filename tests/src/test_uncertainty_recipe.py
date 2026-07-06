"""Mirror test for the uncertainty-propagation recipe of example notebook 14.

The notebook's reverse-route noise propagation relies on ``gamma_extraction_to_infiltration`` being an
exactly linear (not merely affine) map of the extraction signal. That package property is asserted here so
it is maintained as a test rather than only inside the notebook. The weights deliberately do not sum to 1,
so a spurious constant term would be caught as well as genuine nonlinearity.
"""

import numpy as np

from gwtransport import diffusion_fast
from gwtransport.examples import generate_temperature_example_data


def test_reverse_superposition_is_exactly_linear():
    """The deconvolution is an exactly linear map of cout: superposition holds to solver precision."""
    df, tedges = generate_temperature_example_data(
        date_start="2023-01-01",
        date_end="2023-06-01",
        rng=42,
        aquifer_pore_volume_gamma_mean=175.0,
        aquifer_pore_volume_gamma_std=35.0,
        aquifer_pore_volume_gamma_nbins=25,
        longitudinal_dispersivity=0.1,
        molecular_diffusivity=0.01,
        measurement_noise=0.25,
    )

    def reverse(cout):
        cin = diffusion_fast.gamma_extraction_to_infiltration(
            cout=cout,
            flow=df.flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=df.attrs["aquifer_pore_volume_gamma_mean"],
            std=df.attrs["aquifer_pore_volume_gamma_std"],
            n_bins=25,
            retardation_factor=df.attrs["retardation_factor"],
            longitudinal_dispersivity=df.attrs["longitudinal_dispersivity"],
            molecular_diffusivity=df.attrs["molecular_diffusivity"],
            streamline_length=df.attrs["streamline_length"],
            regularization_strength=1e-10,
        )
        return np.asarray(cin)

    cout_a = df.cout.to_numpy()
    cout_b = 0.3 * cout_a + 1.7
    mixed = reverse(0.7 * cout_a + 0.6 * cout_b)
    combo = 0.7 * reverse(cout_a) + 0.6 * reverse(cout_b)
    finite = np.isfinite(mixed) & np.isfinite(combo)
    assert finite.sum() > 100
    residual = np.abs(mixed[finite] - combo[finite]).max() / np.abs(combo[finite]).max()
    assert residual < 1e-9
