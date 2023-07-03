import numpy as np
import pandas as pd

from magnifier.transformer.resampling import change_length, resample


def test_resample_same_rate():
    df = pd.DataFrame(
        {
            "a": np.arange(0.0, 7.0, 2),
        }
    )
    processed_df = resample(df, current_rate=60, target_rate=60)
    assert processed_df.equals(df)


def test_resample_increase_rate():
    df = pd.DataFrame(
        {
            "a": np.arange(0.0, 7.0, 2),
        }
    )
    processed_df = resample(df, current_rate=60, target_rate=120)
    assert processed_df.equals(
        pd.DataFrame(
            {
                "a": np.arange(0.0, 7.0, 1),
            }
        )
    )


def test_resample_decrease_rate():
    df = pd.DataFrame(
        {
            "a": np.arange(0.0, 7.0, 2),
        }
    )
    processed_df = resample(df, current_rate=60, target_rate=20)
    assert processed_df.equals(
        pd.DataFrame(
            {
                "a": np.arange(0.0, 7.0, 6),
            }
        )
    )


def test_change_length_same_length():
    df = pd.DataFrame(
        {
            "a": np.arange(0.0, 7.0, 2),
        }
    )
    processed_df = change_length(df, target_length=4)
    assert len(processed_df) == 4
    assert processed_df.equals(df)


def test_change_length_increase_length():
    df = pd.DataFrame(
        {
            "a": np.arange(0.0, 7.0, 2),
        }
    )
    processed_df = change_length(df, target_length=7)
    assert len(processed_df) == 7
    assert processed_df.equals(
        pd.DataFrame(
            {
                "a": np.arange(0.0, 7.0, 1),
            }
        )
    )


def test_change_length_decrease_length():
    df = pd.DataFrame(
        {
            "a": np.arange(0.0, 7.0, 2),
        }
    )
    processed_df = change_length(df, target_length=2)
    assert len(processed_df) == 2
    assert processed_df.equals(
        pd.DataFrame(
            {
                "a": np.arange(0.0, 7.0, 6),
            }
        )
    )
