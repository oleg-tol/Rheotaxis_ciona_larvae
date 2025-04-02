"""
Brain analysis functions:
  - Phase splitting
  - Preparing custom dataset subsets
  - Grouping cells by brain region and computing summary scores
  - Analyzing datasets by phase
 """

import os
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
from preprocessing import zscore_rawcurves  

def group_cells_to_brain_regions(cell_type):
    group_mapping = {
        'hindbrain': ['mn', 'amg', 'pmg'],
        'midbrain': ['pnsrn', 'prrn', 'antrn', 'ant'],
        'forebrain': ['cor', 'pr'],
        'pns': ['palp', 'rten', 'aten', 'dcen']
    }
    for region, cells in group_mapping.items():
        if cell_type in cells:
            return region
    return np.nan

def prepare_custom_datasets(merged_df):
    subsets = {}
    for strength in ['75', '150', '300']:
        subset = merged_df[merged_df['Stimuli_Strength'] == strength]
        print(f"Stimuli Strength {strength}: {subset.shape[0]} rows")
        subsets[strength] = subset
    return subsets

def split_data_by_phases(df, exposure_time):
    num_timepoints = df.shape[1] - 3  # Exclude metadata columns
    timepoints = [i * exposure_time for i in range(num_timepoints)]
    phases = {
        "before_stimuli": (0, 60),
        "first_stimuli": (60, 90),
        "after_first_stimuli": (90, 150),
        "second_stimuli": (150, 180),
        "after_second_stimuli": (180, num_timepoints * exposure_time)
    }
    phase_data = {}
    for phase, (start, end) in phases.items():
        indices = [i + 1 for i, t in enumerate(timepoints) if start <= t < end]
        phase_data[phase] = df.iloc[:, [0] + indices] 
    return phase_data

def calculate_weighted_summary(df_loadings, explained_variance, use_abs=True):
    df_loadings_transposed = df_loadings.T
    n_components = len(explained_variance)
    weighted_loadings = df_loadings_transposed.multiply(explained_variance, axis=1)
    summary_score = weighted_loadings.abs().sum(axis=1) if use_abs else weighted_loadings.sum(axis=1)
    return summary_score.round(3)

def generate_and_save_summary_data(subsets, output_dir, exposure_time):
    summary_data = {}
    all_cell_types = ['mn', 'amg', 'pmg', 'pnsrn', 'prrn', 'antrn', 'ant', 'cor', 'pr', 'palp', 'rten', 'aten', 'dcen']
    
    for strength, subset_df in subsets.items():
        phase_data = split_data_by_phases(subset_df, exposure_time)
        for phase, phase_df in phase_data.items():
            df_zscored = zscore_rawcurves(phase_df)
           
            from pca_analysis import get_PCA_results
            results_pca, df_loadings, explained_variance = get_PCA_results(df_zscored, n_components=5)
            if results_pca is None or df_loadings is None:
                print(f"Skipping PCA for phase {phase} due to insufficient data for strength {strength}.")
                continue
            individual_summary_scores = calculate_weighted_summary(df_loadings, explained_variance, use_abs=True)
            individual_summary_scores = individual_summary_scores.to_dict()
            for cell_type in all_cell_types:
                if cell_type not in individual_summary_scores:
                    individual_summary_scores[cell_type] = np.nan
            phase_df['brain_region'] = phase_df['cell'].apply(group_cells_to_brain_regions)
            phase_df = phase_df.dropna(subset=['brain_region'])
            for region in ['hindbrain', 'midbrain', 'forebrain', 'pns']:
                summary_data.setdefault(region, {}).setdefault(strength, {})[phase] = np.nan
            for region in phase_df['brain_region'].unique():
                cells_in_region = phase_df[phase_df['brain_region'] == region]['cell']
                region_score = np.nansum([individual_summary_scores.get(cell, np.nan) for cell in cells_in_region])
                summary_data[region][strength][phase] = np.round(region_score, 3)
    return summary_data

def print_cell_type_counts(subsets):
    for strength, subset_df in subsets.items():
        print(f"\nStimuli Strength {strength}:")
        print(subset_df['cell'].value_counts().to_string())
