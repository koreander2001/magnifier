from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from python_speech_features import mfcc
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
    scalers: List[StandardScaler] = field(default_factory=list)

    def fit(self, X, y=None) -> "StandardScaler3d":
        self.scalers = list(
            map(lambda i: StandardScaler().fit(X[:, i, :]), range(X.shape[1]))
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_result = np.copy(X)

        for i, scaler in enumerate(self.scalers):
            X_result[:, i, :] = scaler.transform(X[:, i, :])

        return X_result
