"""
Module for calculating various experimental metrics.
Includes functions for:
  - Synchronizing NaNs between paired dataframes
  - Calculate center of mass
  - Filtering continuous data fragments
  - Calculating heading velocity
  - Applying pixel size conversion to numerical data
  - Filtering speeds by threshold
  - Calculating Euclidean distance between frames
  - Calculating angles (in radians) from paired coordinates
  - Computing probability of angle values within bounds over time windows
  - Calculating angular velocity (omega)
  - Computing Mean Squared Displacement (MSD) using a sliding window
"""

import numpy as np
import pandas as pd

def synchronize_nans(x_df, y_df):
    """Set mismatches between x and y as NaN."""
    nan_mask = x_df.isna() | y_df.isna()
    return x_df.where(~nan_mask, np.nan), y_df.where(~nan_mask, np.nan)

def calculate_center_of_mass(df_b, df_p):
    if df_b.shape != df_p.shape:
        raise ValueError("DataFrames must have the same shape")
    com_df = pd.DataFrame(index=df_b.index, columns=df_b.columns)
    
    for col in df_b.columns:
        b_values = np.nan_to_num(df_b[col])
        p_values = np.nan_to_num(df_p[col])
        com_df[col] = (b_values + p_values) / 2

    com_df = com_df.replace(0, np.nan)
    return com_df


def filter_continuous_fragments(df, min_length=10):
    """Return only fragments with at least min_length continuous non-NaN frames."""
    filtered_df = pd.DataFrame(index=df.index, columns=df.columns)
    for i, row in df.iterrows():
        valid = row.notna()
        fragments = valid.astype(int).groupby((valid != valid.shift()).cumsum()).transform('sum')
        mask = fragments >= min_length
        filtered_df.loc[i] = row.where(mask, np.nan)
    return filtered_df

def calculate_heading_velocity(x_df, y_df, exposure_time, col_start=2):
    """
    Compute heading velocity using differences in x and y coordinates.
    Assumes numeric columns start at index col_start.
    """
    dx = x_df.iloc[:, col_start:-1].diff(axis=1)
    dy = y_df.iloc[:, col_start:-1].diff(axis=1)
    velocity = np.sqrt(dx**2 + dy**2) / exposure_time
    return velocity.where(~(dx.isna() | dy.isna()))

def apply_pixel_size(df, pixel_size, cat_cols=2):
    """
    Multiply numerical columns (from index cat_cols onward) by pixel_size.
    Categorical columns (first cat_cols columns) remain unchanged.
    """
    categorical = df.iloc[:, :cat_cols]
    numerical = df.iloc[:, cat_cols:] * pixel_size
    return pd.concat([categorical.reset_index(drop=True), numerical.reset_index(drop=True)], axis=1)

def filter_speeds(df, threshold, cat_cols=2):
    """
    Set values below the threshold in numerical columns to NaN.
    """
    categorical = df.iloc[:, :cat_cols]
    numerical = df.iloc[:, cat_cols:].where(df.iloc[:, cat_cols:] >= threshold)
    return pd.concat([categorical.reset_index(drop=True), numerical.reset_index(drop=True)], axis=1)

def calculate_distance(x_df, y_df, col_start=2):
    """
    Calculate Euclidean distance between consecutive frames.
    """
    dx = x_df.iloc[:, col_start:-1].diff(axis=1)
    dy = y_df.iloc[:, col_start:-1].diff(axis=1)
    distance = np.sqrt(dx**2 + dy**2)
    return distance.where(~(dx.isna() | dy.isna()))

def calculate_angles(x_p, y_p, x_b, y_b, offset=0.061, col_start=2):
    """
    Calculate angles (radians) between points with an offset.
    Normalizes angles to [0, 2π].
    """
    angles = np.arctan2(
        x_p.iloc[:, col_start:] - x_b.iloc[:, col_start:],
        y_p.iloc[:, col_start:] - y_b.iloc[:, col_start:]
    ) + offset
    return np.mod(angles, 2 * np.pi)

def calculate_probability(angles_df, angle_lower_deg, angle_upper_deg, time_window_size,
                          cat_columns=['Experiment', 'Stimulus_Strength', 'Count_cat']):
    """
    Calculate the mean probability per time window that angle values fall within the specified bounds.
    Returns a DataFrame with one row of window probabilities and reattached categorical info.
    """
    time_cols = [col for col in angles_df.columns if col not in cat_columns]
    num_timepoints = len(time_cols)
    num_windows = num_timepoints // time_window_size
    prob_list = []
    for i in range(num_windows):
        start = i * time_window_size
        end = start + time_window_size
        window_data = angles_df[time_cols[start:end]]
        # Calculate per-column probability, then average over the window
        prob = window_data.apply(
            lambda col: np.nan if col.isna().all() else np.sum((col >= angle_lower_deg) & (col <= angle_upper_deg)) / col.notna().sum(),
            axis=0
        ).mean()
        prob_list.append(prob)
    prob_df = pd.DataFrame([prob_list], columns=[f'Window_{i}' for i in range(num_windows)])
    for col in cat_columns:
        prob_df[col] = angles_df[col].iloc[0]
    return prob_df

def calculate_omega(angles_df, dt, col_start=1):
    """
    Compute angular velocity (omega) as the difference in angles divided by dt.
    Expects numeric angle columns starting at col_start.
    """
    omega_values = (angles_df.iloc[:, col_start:].diff(axis=1)) / dt
    omega_values = omega_values.astype(float)
    return omega_values[~omega_values.isna().all(axis=1)]

def calculate_single_msd(com_x, com_y):
    """
    Compute Mean Squared Displacement (MSD) using a sliding window approach.
    Returns two arrays: mean MSD and standard error for each window.
    """
    N = com_x.shape[1]
    max_window = N // 2
    SW = np.zeros(max_window)
    SWe = np.zeros(max_window)
    for wind in range(1, max_window + 1):
        displacements = []
        for t in range(wind, N):
            dx = com_x.iloc[:, t] - com_x.iloc[:, t - wind]
            dy = com_y.iloc[:, t] - com_y.iloc[:, t - wind]
            d2 = dx**2 + dy**2
            displacements.extend(d2.dropna().values)
        if displacements:
            SW[wind - 1] = np.mean(displacements)
            SWe[wind - 1] = np.std(displacements) / np.sqrt(len(displacements))
        else:
            SW[wind - 1] = np.nan
            SWe[wind - 1] = np.nan
    return SW, SWe
