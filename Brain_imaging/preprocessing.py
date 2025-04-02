
"""
Functios for preprocessing and z-score normalization of raw calcium imaging time-series data.
"""

import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

def zscore_rawcurves(sample_df):
   try:
        # Exclude metadata columns; assume first 3 columns are metadata: 'cell', 'Experiment', 'Stimuli_Strength'
        temporal = sample_df.iloc[:, 3:].values.astype(float)
        zscored = TimeSeriesScalerMeanVariance().fit_transform(temporal)[..., 0]
        df_zscored = pd.DataFrame(zscored, index=sample_df.index, columns=sample_df.columns[3:])
        df_zscored.insert(0, 'cell', sample_df['cell'])
        return df_zscored
    except Exception as e:
        print(f"Error in z-score normalization: {e}")
        return pd.DataFrame()
