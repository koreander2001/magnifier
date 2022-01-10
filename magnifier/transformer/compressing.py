from dataclasses import dataclass

import numpy as np

from ..base import BaseTransformer


@dataclass
class MeanCompressor(BaseTransformer):
    width: int = int()

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._compress, axis=1, arr=X)

    def _compress(self, X: np.ndarray) -> np.ndarray:
        convolved_arr = np.convolve(X, np.ones(10) / self.width, mode="same")
        return convolved_arr[np.arange(0, convolved_arr.shape[0], self.width)]
