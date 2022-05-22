import re

import numpy as np
import pytest
from magnifier.transformer.time_series import SlidingWindow


class TestSlidingWindow:
    def test_init_exception(self):
        width = 3.14
        stepsize = 1
        error_message = f"`width` is not `int`, width: {width}."
        with pytest.raises(TypeError, match=error_message):
            _ = SlidingWindow(width=width, stepsize=stepsize)

        width = 5
        stepsize = 3.14
        error_message = f"`stepsize` is not `int`, stepsize: {stepsize}."
        with pytest.raises(TypeError, match=error_message):
            _ = SlidingWindow(width=width, stepsize=stepsize)

        width = 0
        stepsize = 10
        error_message = f"`width` is not positive number, width: {width}."
        with pytest.raises(ValueError, match=error_message):
            _ = SlidingWindow(width=width, stepsize=stepsize)

        width = 10
        stepsize = 0
        error_message = f"`stepsize` is not positive number, stepsize: {stepsize}."
        with pytest.raises(ValueError, match=error_message):
            _ = SlidingWindow(width=width, stepsize=stepsize)

        width = 9
        stepsize = 10
        error_message = (
            f"`width` less than `stepsize`, width: {width}, stepsize: {stepsize}."
        )
        with pytest.raises(ValueError, match=error_message):
            _ = SlidingWindow(width=width, stepsize=stepsize)

    def test_transform_exception(self):
        width = 9
        stepsize = 1
        sliding_window = SlidingWindow(width=width, stepsize=stepsize)

        X = [0, 1, 2, 3, 4, 5, 6, 7]
        error_message = f"Type of X must be np.ndarray, but given: {type(X)}."
        with pytest.raises(TypeError, match=error_message):
            _ = sliding_window.transform(X)

        X = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        error_message = (
            f"`X.shape[-1]` less than `width`, width: {width}, shape of X: {X.shape}."
        )
        with pytest.raises(ValueError, match=re.escape(error_message)):
            _ = sliding_window.transform(X)

    def test_transform_1_dim(self):
        X = np.arange(6)

        sliding_window = SlidingWindow(width=4, stepsize=2)
        output_arr = sliding_window.transform(X)

        expected_arr = np.array(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
            ]
        )
        assert np.all(output_arr == expected_arr)

    def test_transform_1_dim_with_truncated_data(self):
        X = np.arange(7)

        sliding_window = SlidingWindow(width=4, stepsize=2)
        output_arr = sliding_window.transform(X)

        expected_arr = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
        assert np.all(output_arr == expected_arr)

    def test_transform_not_1_dim(self):
        X = np.array(
            [
                [
                    [0, 1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12, 13],
                ],
                [
                    [14, 15, 16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25, 26, 27],
                ],
            ]
        )

        sliding_window = SlidingWindow(width=4, stepsize=2)
        output_arr = sliding_window.transform(X)

        expected_arr = np.array(
            [
                [
                    [[0, 1, 2, 3], [2, 3, 4, 5]],
                    [[7, 8, 9, 10], [9, 10, 11, 12]],
                ],
                [
                    [[14, 15, 16, 17], [16, 17, 18, 19]],
                    [[21, 22, 23, 24], [23, 24, 25, 26]],
                ],
            ]
        )
        assert np.all(output_arr == expected_arr)
