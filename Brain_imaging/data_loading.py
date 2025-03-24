"""
Functions to load and merge CSV files containing calcium imaging data.
"""

import os
import re
import pandas as pd
from glob import glob
from datetime import date

def load_data(directory, pattern="*.csv", verbose=True):
    """
    Load and merge CSV files from the specified directory.
    
    Parameters:
        directory (str): Path to the directory containing CSV files.
        pattern (str): Pattern to match CSV files (default is '*.csv').
        verbose (bool): If True, print progress messages.

    Returns:
        pd.DataFrame: Merged DataFrame containing data from all CSV files.
    """
    files = glob(os.path.join(directory, pattern))
    if verbose:
        print(f"Found {len(files)} files in {directory}.")

    merged_df = pd.DataFrame()
    for file in files:
        try:
            filename = os.path.basename(file)
            experiment = filename[:-4]  # Remove '.csv'
            stimuli_part = re.search(r'Gcamp6-(\d+)', filename)
            stimuli_strength = stimuli_part.group(1) if stimuli_part else 'Unknown'

            df = pd.read_csv(file, index_col=0)
            df['Experiment'] = experiment
            df['Stimuli_Strength'] = stimuli_strength

            merged_df = pd.concat([merged_df, df], ignore_index=True)
            if verbose:
                print(f"Loaded {filename} with {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    output_dir = os.path.join(directory, "Output_new")
    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, "merged_data.csv")
    merged_df.round(3).to_csv(csv_output_path, index=False)

    if verbose:
        print(f"Merged data saved to {csv_output_path}.")
        print(f"DataFrame Shape: {merged_df.shape}")
        print(f"Columns: {merged_df.columns.tolist()}")

    return merged_df
