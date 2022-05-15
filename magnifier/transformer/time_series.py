from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from python_speech_features import mfcc
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from ..base import BaseTransformer


@dataclass
class MFCC(BaseTransformer):
    """MFCCを適用する"""

    frame_rate: int
    winlen: float = 0.025
    winstep: float = 0.01
    numcep: int = 13
    nfft: Optional[int] = None
    winfunc: Callable[[int], np.ndarray] = np.hamming

    def transform(self, X):
        return np.apply_along_axis(self._mfcc, axis=1, arr=X.astype(float))

    def _mfcc(self, X):
        nfft = self.nfft if self.nfft else X.size * 2

        return mfcc(
            X,
            samplerate=self.frame_rate,
            winlen=self.winlen,
            winstep=self.winstep,
            numcep=self.numcep,
            nfilt=self.numcep * 2,
            nfft=nfft,
            winfunc=self.winfunc,
        )


@dataclass
class SlidingWindow(BaseTransformer):
    width: int
    stepsize: int

    def __post_init__(self):
        self._check_parameter()

    def transform(self, X):
        self._check_X(X)

        data_size = X.shape[-1]

        index_ranges = []
        for start in range(0, data_size, self.stepsize):
            end = start + self.width
            if end > data_size:
                break
            index_ranges.append((start, end))

        return np.apply_along_axis(self._extract_by_index_ranges, -1, X, index_ranges)

    def _check_parameter(self):
        if not isinstance(self.width, int):
            raise TypeError(f"`width` is not `int`, width: {self.width}.")
        if not isinstance(self.stepsize, int):
            raise TypeError(f"`stepsize` is not `int`, stepsize: {self.stepsize}.")

        if self.width < 1:
            raise ValueError(f"`width` is not positive number, width: {self.width}.")
        if self.stepsize < 1:
            raise ValueError(
                f"`stepsize` is not positive number, stepsize: {self.stepsize}."
            )

        if self.width < self.stepsize:
            raise ValueError(
                f"`width` less than `stepsize`, width: {self.width}, stepsize: {self.stepsize}."
            )

    def _check_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Type of X must be np.ndarray, but given: {type(X)}.")

        if X.shape[-1] < self.width:
            raise ValueError(
                f"`X.shape[-1]` less than `width`, width: {self.width}, shape of X: {X.shape}."
            )

    def _extract_by_index_ranges(
        self, X: np.ndarray, index_ranges: List[Tuple[int, int]]
    ) -> np.ndarray:
        return np.array([X[start:end] for start, end in index_ranges])


@dataclass
class StandardScaler3d(BaseTransformer):
    _scalers: List[StandardScaler] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: None = None) -> "StandardScaler3d":
        self._check_X(X)

        self._scalers = list(
            map(
                lambda i: StandardScaler().fit(X[:, :, i].reshape(-1, 1)),
                range(X.shape[2]),
            )
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_X_on_transform(X)

        X_result = np.copy(X)

        for i, scaler in enumerate(self._scalers):
            X_result[:, :, i] = scaler.transform(X[:, :, i].reshape(-1, 1)).reshape(
                X.shape[:-1]
            )

        return X_result

    def _check_X(self, X: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Type of X must be np.ndarray, but given: {type(X)}")
        if X.ndim != 3:
            raise ValueError(
                f"Number of dimensions of X must be 3, but given: {X.shape}."
            )

    def _check_X_on_transform(self, X: np.ndarray) -> None:
        if not self._scalers:
            raise NotFittedError(f"{type(self).__name__} is not fitted.")

        self._check_X(X)

        if X.shape[2] != len(self._scalers):
            raise ValueError(
                f"X.shape[1] must be equal to {len(self._scalers)}, but given: {X.shape}."
            )
