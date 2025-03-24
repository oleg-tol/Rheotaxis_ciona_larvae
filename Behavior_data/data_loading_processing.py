"""
Module for loading, processing, and filtering (based on body size of the animal) DLC output filtered CSV data files.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_output_folder(base_folder: str) -> str:
    base_output_path = os.path.join(base_folder, "Processed_Results")
    output_folder = os.path.join(base_output_path, datetime.today().strftime('%Y-%m-%d'))
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")
    return output_folder

def process_csv_file(file: str, output_folder: str, skipped_files: list, pixel_size: int = 25,
                     rate: int = 5, distance_threshold_lower: int = 100, distance_threshold_upper: int = 500):
    try:
        with open(file, 'r', encoding='ISO-8859-1') as f:
            first_line = f.readline()
            total_columns = len(first_line.split(',')) - 1

        usecols_range = range(1, total_columns + 1)
        df = pd.read_csv(file, header=None, skiprows=1, usecols=usecols_range,
                         low_memory=False, encoding='ISO-8859-1')
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        dt = df.T.copy()

        x_p = dt[(dt[1] == "palp") & (dt[2] == "x")].iloc[:, 3:].T
        y_p = dt[(dt[1] == "palp") & (dt[2] == "y")].iloc[:, 3:].T
        x_b = dt[(dt[1] == "backtrunk") & (dt[2] == "x")].iloc[:, 3:].T
        y_b = dt[(dt[1] == "backtrunk") & (dt[2] == "y")].iloc[:, 3:].T

        individual_ids = dt[0].unique()
        print(f"Processing file: {file}")
        print(f"Number of individuals: {len(individual_ids)}")
        x_p.columns = individual_ids
        y_p.columns = individual_ids
        x_b.columns = individual_ids
        y_b.columns = individual_ids

        x_p_sampled = x_p[::rate].astype(float)
        y_p_sampled = y_p[::rate].astype(float)
        x_b_sampled = x_b[::rate].astype(float)
        y_b_sampled = y_b[::rate].astype(float)

        distances_b_p = {}
        for col in individual_ids:
            curr_x_p = pd.to_numeric(x_p_sampled[col], errors='coerce')
            curr_y_p = pd.to_numeric(y_p_sampled[col], errors='coerce')
            curr_x_b = pd.to_numeric(x_b_sampled[col], errors='coerce')
            curr_y_b = pd.to_numeric(y_b_sampled[col], errors='coerce')
            dist_pb = np.sqrt((curr_x_p - curr_x_b) ** 2 + (curr_y_p - curr_y_b) ** 2) * pixel_size
            distances_b_p[col] = dist_pb

        distances_b_p_df = pd.DataFrame(distances_b_p)
        mask = (distances_b_p_df >= distance_threshold_lower) & (distances_b_p_df <= distance_threshold_upper)

        x_p_sampled = x_p_sampled.where(mask, np.nan).replace(0, np.nan)
        y_p_sampled = y_p_sampled.where(mask, np.nan).replace(0, np.nan)
        x_b_sampled = x_b_sampled.where(mask, np.nan).replace(0, np.nan)
        y_b_sampled = y_b_sampled.where(mask, np.nan).replace(0, np.nan)

        base_name = os.path.splitext(os.path.basename(file))[0]
        print(f"Saving processed files for {base_name} to {output_folder}")
        x_p_sampled.to_csv(os.path.join(output_folder, f'{base_name}_sampled_X_p.csv'))
        y_p_sampled.to_csv(os.path.join(output_folder, f'{base_name}_sampled_Y_p.csv'))
        x_b_sampled.to_csv(os.path.join(output_folder, f'{base_name}_sampled_X_b.csv'))
        y_b_sampled.to_csv(os.path.join(output_folder, f'{base_name}_sampled_Y_b.csv'))
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        skipped_files.append(file)

def process_experiment_folder(experiment_folder: str):
    skipped_files = []
    output_folder = create_output_folder(experiment_folder)
    for root, dirs, files in os.walk(experiment_folder):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            print("Processing folder:", folder_path)
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('_filtered.csv')]
            for csv_file in csv_files:
                csv_file_path = os.path.join(folder_path, csv_file)
                print("Processing CSV file:", csv_file)
                process_csv_file(csv_file_path, output_folder, skipped_files)
    if skipped_files:
        print("\nThe following files were skipped due to errors:")
        for file in skipped_files:
            print(file)
    else:
        print("\nAll files processed successfully!")

if __name__ == '__main__':
    experiment_folder = '/media/oleg/oleg/ChH_new'
    process_experiment_folder(experiment_folder)
