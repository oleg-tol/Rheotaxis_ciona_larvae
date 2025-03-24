"""
Functions to perform PCA on z-scored data and to compute PCA on the entire dataset.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def get_PCA_results(df_zscored, n_components=3):
    """
    Perform PCA on z-scored data and return PCA results and loadings.
    
    Parameters:
        df_zscored (pd.DataFrame): Z-scored data.
        n_components (int): Number of PCA components.
        
    Returns:
        Tuple: (PCA transformed data, PCA loadings DataFrame, explained variance ratio)
    """
    try:
        zscored = df_zscored.values
        n_components = min(zscored.shape[0], zscored.shape[1], n_components)
        if n_components < 2:
            print(f"Insufficient components for PCA. Required >= 2, got {n_components}.")
            return None, None, None

        pca = PCA(n_components=n_components)
        results_pca = pca.fit_transform(zscored)
        loadings = pca.components_
        df_loadings = pd.DataFrame(loadings, columns=df_zscored.columns)
        explained_variance = pca.explained_variance_ratio_ * 100

        for i, variance in enumerate(explained_variance):
            print(f"PC{i+1} explains {variance:.2f}% of the variance.")
        
        return results_pca, df_loadings, explained_variance
    except Exception as e:
        print(f"Error in PCA computation: {e}")
        return None, None, None

def perform_pca_on_entire_dataset(merged_df, n_components=3):
    """
    Perform PCA on the entire dataset (without phase separation) and save eigenvectors.
    
    Parameters:
        merged_df (pd.DataFrame): The entire dataset.
        n_components (int): Number of principal components.
        
    Returns:
        Tuple: (PCA object, explained variance ratio)
    """
    try:
        # Assume data columns are from 1 to -2 (exclude metadata)
        data_columns = merged_df.columns[1:-2]
        entire_data = merged_df[data_columns].values
        zscored_data = TimeSeriesScalerMeanVariance().fit_transform(entire_data)[..., 0]
        pca = PCA(n_components=n_components)
        pca.fit(zscored_data)
        explained_variance = pca.explained_variance_ratio_ * 100
        for i, variance in enumerate(explained_variance):
            print(f"PC{i+1} explains {variance:.2f}% of the variance.")
        eigenvectors_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(n_components)])
        eigenvectors_df.index = data_columns
        return pca, explained_variance
    except Exception as e:
        print(f"Error performing PCA on entire dataset: {e}")
        return None, None
