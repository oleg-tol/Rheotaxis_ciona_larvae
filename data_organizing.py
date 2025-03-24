#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:44:01 2025

@author: oleg
"""
"""
data_preprocessing.py

Module for organizing, validating, and merging processed CSV files from stimulus folders.
"""

import os
import glob
import pandas as pd
import numpy as np

def validate_and_sort(files, expected=None, label=""):
    files = sorted(files)
    if expected is not None and len(files) != expected:
        print(f"[WARNING] Expected {expected} {label} files but found {len(files)}.")
    return files

def process_stimulus_folder(stimulus_folder):
    strength = os.path.basename(stimulus_folder)
    files_X_b = validate_and_sort(glob.glob(os.path.join(stimulus_folder, '**/*_sampled_X_b.csv'), recursive=True))
    files_Y_b = validate_and_sort(glob.glob(os.path.join(stimulus_folder, '**/*_sampled_Y_b.csv'), recursive=True), expected=len(files_X_b), label="Y_b")
    files_X_p = validate_and_sort(glob.glob(os.path.join(stimulus_folder, '**/*_sampled_X_p.csv'), recursive=True), expected=len(files_X_b), label="X_p")
    files_Y_p = validate_and_sort(glob.glob(os.path.join(stimulus_folder, '**/*_sampled_Y_p.csv'), recursive=True), expected=len(files_X_b), label="Y_p")

    if not (len(files_X_b) == len(files_Y_b) == len(files_X_p) == len(files_Y_p)):
        print(f"[ERROR] File mismatch in {strength}, skipping.")
        return [], [], [], []

    dfs_X_b, dfs_Y_b, dfs_X_p, dfs_Y_p = [], [], [], []
    for fXb, fYb, fXp, fYp in zip(files_X_b, files_Y_b, files_X_p, files_Y_p):
        exp = os.path.basename(fXb).split('_sampled')[0]
        try:
            df_X_b = pd.read_csv(fXb).T.iloc[1:]
            df_Y_b = pd.read_csv(fYb).T.iloc[1:]
            df_X_p = pd.read_csv(fXp).T.iloc[1:]
            df_Y_p = pd.read_csv(fYp).T.iloc[1:]
            if not (len(df_X_b) == len(df_Y_b) == len(df_X_p) == len(df_Y_p)):
                print(f"[WARNING] Data length mismatch in {exp}, skipping.")
                continue
            for df in (df_X_b, df_Y_b, df_X_p, df_Y_p):
                df['Experiment'], df['Stimulus_Strength'] = exp, strength
            dfs_X_b.append(df_X_b)
            dfs_Y_b.append(df_Y_b)
            dfs_X_p.append(df_X_p)
            dfs_Y_p.append(df_Y_p)
        except Exception as e:
            print(f"[ERROR] Failed {exp}: {e}")
    return dfs_X_b, dfs_Y_b, dfs_X_p, dfs_Y_p

def merge_and_reorder(dfs):
    if not dfs:
        return pd.DataFrame()
    merged = pd.concat(dfs, ignore_index=True)
    for col in ['Experiment', 'Stimulus_Strength']:
        if col in merged.columns:
            merged = merged[[col] + [c for c in merged.columns if c != col]]
    return merged

def edge_cut(df, x_min=100, x_max=1100):
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce') \
                                 .applymap(lambda x: x if np.isnan(x) or x_min <= x <= x_max else np.nan)
    return df

def prepare_df(df):
    numeric = df.iloc[:, 2:].astype(float).dropna(how='all')
    return pd.concat([df[['Experiment', 'Stimulus_Strength']], numeric], axis=1)

def process_all(parent_dir):
    folders = glob.glob(os.path.join(parent_dir, '*'))
    all_X_b, all_Y_b, all_X_p, all_Y_p = [], [], [], []
    for folder in folders:
        dfs = process_stimulus_folder(folder)
        all_X_b += dfs[0]
        all_Y_b += dfs[1]
        all_X_p += dfs[2]
        all_Y_p += dfs[3]
    merged_X_b = prepare_df(edge_cut(merge_and_reorder(all_X_b)))
    merged_Y_b = prepare_df(merge_and_reorder(all_Y_b))
    merged_X_p = prepare_df(edge_cut(merge_and_reorder(all_X_p)))
    merged_Y_p = prepare_df(merge_and_reorder(all_Y_p))
    return merged_X_b, merged_Y_b, merged_X_p, merged_Y_p

def main():
    parent_dir = '/media/oleg/oleg/ChH_new/Processed_Results'
    output_dir = '/media/oleg/oleg/ChH_new/Processed_Results/2025-03-24/Results'
    os.makedirs(output_dir, exist_ok=True)
    df_X_b, df_Y_b, df_X_p, df_Y_p = process_all(parent_dir)
    df_X_b.to_csv(os.path.join(output_dir, 'filtered_df_X_b.csv'), index=False)
    df_Y_b.to_csv(os.path.join(output_dir, 'filtered_df_Y_b.csv'), index=False)
    df_X_p.to_csv(os.path.join(output_dir, 'filtered_df_X_p.csv'), index=False)
    df_Y_p.to_csv(os.path.join(output_dir, 'filtered_df_Y_p.csv'), index=False)
    print("Data processing complete.")

if __name__ == '__main__':
    main()

