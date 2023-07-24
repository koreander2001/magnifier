import numpy as np

from magnifier.eda.correlation import xcorr


def test_xcorr():
    a: np.ndarray = np.sin(np.arange(3600) * np.pi / 180.0)
    v = a.copy()

    corr_arr = xcorr(a, v)

    assert np.all((-1.000001 < corr_arr) & (corr_arr < 1.000001))

    no_lag_index = a.size - 1
    np.testing.assert_almost_equal(corr_arr[no_lag_index], 1.0)
    assert corr_arr[no_lag_index - 180] < -0.95
    assert corr_arr[no_lag_index + 180] < -0.95
