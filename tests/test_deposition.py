import numpy as np
import pandas as pd
import pytest

from gwtransport1d.deposition import compute_deposition, compute_dc, deposition_coefficients


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    cout = pd.Series(data=np.random.rand(len(dates)), index=dates)
    flow = pd.Series(data=np.random.rand(len(dates)) * 100, index=dates)
    deposition = pd.Series(data=np.random.rand(len(dates)), index=dates)
    return cout, flow, deposition


def test_compute_deposition(sample_data):
    cout, flow, _ = sample_data
    aquifer_pore_volume = 200.0
    porosity = 0.3
    thickness = 15.0
    retardation_factor = 2.1

    deposition = compute_deposition(
        cout, flow, aquifer_pore_volume, porosity, thickness, retardation_factor
    )

    assert isinstance(deposition, pd.Series)
    assert len(deposition) == len(cout)
    assert np.all(deposition >= 0)


def test_compute_dc(sample_data):
    cout, flow, deposition = sample_data
    aquifer_pore_volume = 200.0
    porosity = 0.3
    thickness = 15.0
    retardation_factor = 2.1

    dcout = compute_dc(
        cout.index, deposition, flow, aquifer_pore_volume, porosity, thickness, retardation_factor
    )

    assert isinstance(dcout, pd.Series)
    assert len(dcout) == len(cout)
    assert np.all(dcout >= 0)


def test_deposition_coefficients(sample_data):
    cout, flow, _ = sample_data
    aquifer_pore_volume = 200.0
    porosity = 0.3
    thickness = 15.0
    retardation_factor = 2.1

    coeff, df, index_dep = deposition_coefficients(
        cout.index, flow, aquifer_pore_volume, porosity, thickness, retardation_factor
    )

    assert isinstance(coeff, np.ndarray)
    assert coeff.shape[0] == len(cout)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(index_dep, pd.DatetimeIndex)
    assert len(index_dep) >= len(cout)
    assert np.isclose(coeff.sum(axis=0), 1.0, rtol=1e-5).all()
    assert np.isclose(coeff.sum(axis=1), 1.0, rtol=1e-5).all()
