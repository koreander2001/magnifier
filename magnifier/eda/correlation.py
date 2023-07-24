import numpy as np


def xcorr(a: np.ndarray, v: np.ndarray, normed=True) -> np.ndarray:
    xcorr_arr = np.correlate(a, v, mode="full")

    if normed:
        xcorr_arr /= np.linalg.norm(a, ord=2) * np.linalg.norm(v, ord=2)

    return xcorr_arr
