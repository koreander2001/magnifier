import numpy as np
import pandas as pd


def resample(
    data: pd.DataFrame, current_rate: int, target_rate: int, temp_col: str = "temp_col"
) -> pd.DataFrame:
    lcm_rate = np.lcm(current_rate, target_rate)

    time_df = data.copy()
    time_df[temp_col] = pd.to_datetime(
        np.arange(len(data)) * (lcm_rate // current_rate), unit="s"
    )
    time_df = time_df.set_index(temp_col)

    resampled_df = (
        time_df.resample("1s")
        .interpolate()
        .asfreq(f"{lcm_rate // target_rate}s")
        .reset_index(drop=True)
    )
    return resampled_df


def change_length(data: pd.DataFrame, target_length: int) -> pd.DataFrame:
    current_length = data.shape[0]
    return resample(data, current_length - 1, target_length - 1)
