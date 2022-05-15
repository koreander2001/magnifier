from dataclasses import dataclass, field
from typing import Callable, ClassVar, List, Optional

import numpy as np
from python_speech_features import mfcc
from scipy.stats import kurtosis, skew
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
class StandardScaler3d(BaseTransformer):
    _scalers: List[StandardScaler] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: None = None, **fit_params) -> "StandardScaler3d":
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


@dataclass
class Summarizer(BaseTransformer):
    """TODO: Add description"""

    ALL_STATS: ClassVar[List[str]] = (
        "min",
        "max",
        "mean",
        "std",
        "var",
        "skew",
        "kurt",
    )

    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    _target_stats: List[str] = field(init=False)

    def __post_init__(self) -> None:
        _target_stats = set(self.include or self.ALL_STATS) - set(self.exclude)
        self._target_stats = list(filter(_target_stats.__contains__, self.ALL_STATS))

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        stats_list = []
        if "min" in self._target_stats:
            stats_list.append(X.min(axis=2).reshape(X.shape[0], X.shape[1], 1))
        if "max" in self._target_stats:
            stats_list.append(X.max(axis=2).reshape(X.shape[0], X.shape[1], 1))
        if "mean" in self._target_stats:
            stats_list.append(X.mean(axis=2).reshape(X.shape[0], X.shape[1], 1))
        if "std" in self._target_stats:
            stats_list.append(X.std(axis=2).reshape(X.shape[0], X.shape[1], 1))
        if "var" in self._target_stats:
            stats_list.append(X.var(axis=2).reshape(X.shape[0], X.shape[1], 1))
        if "skew" in self._target_stats:
            stats_list.append(skew(X, axis=2).reshape(X.shape[0], X.shape[1], 1))
        if "kurt" in self._target_stats:
            stats_list.append(kurtosis(X, axis=2).reshape(X.shape[0], X.shape[1], 1))

        return np.concatenate(stats_list, axis=2)
