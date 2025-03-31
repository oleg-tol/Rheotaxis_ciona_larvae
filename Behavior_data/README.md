# Deep Lab Cut experimental data processing

This repository contains two modules for processing experimental data, and one module for behavior metrics calculation.

## Modules

- **data_loading_processing.py**
  Loads and processes raw CSV files by cleaning, sampling, and filtering based on known animal size.

- **data_organizing.py**
  Organizes and merges processed CSV files from different stimulus folders, applies additional filtering (edge cutting), and reorders metadata columns.

## Requirements

- Python 3.x
- Pandas
- NumPy

## Setup

1. Install the required packages.
2. Update file paths as needed in the source files.
3. Run the desired module.

# Metric Calculation Module

This module (behavior_metrics.py) contains a collection of functions to compute the key
metrics used in our experimental analyses. The functions include:

- synchronize_nans: Synchronize NaNs between paired dataframes.
- calculate_center_of_mass : Calculate the center of mass from the body parts coordinates.
- filter_continuous_fragments: Filter data to keep only continuous fragments with at least a minimum length.
- calculate_heading_velocity: Compute heading velocity from x and y coordinates.
- apply_pixel_size: Convert numerical data using a pixel size factor.
- filter_speeds: Filter out values below a specified threshold.
- calculate_distance: Compute the Euclidean distance between consecutive frames.
- calculate_angles: Calculate angles (in radians) between paired coordinates.
- calculate_probability: Compute the probability that angle values fall within specified bounds over time windows.
- calculate_omega: Calculate angular velocity (omega) from angle time series.
- calculate_single_msd: Compute Mean Squared Displacement (MSD) using a sliding window approach.

## Usage

Import the module in your script to access the functions.
