import os
import pandas as pd
import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from statsmodels.tsa.seasonal import seasonal_decompose

def read_and_prepare(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    filename = os.path.basename(file)
    for col in df.columns:
        if 'Mean' in col:
            df[col] = df[col].apply(
                lambda x: [float(y.replace('.', '').replace(',', '.')) for y in x.split('\t')]
                if isinstance(x, str) else ([float(x)] if not pd.isna(x) else [0.0])
            )
            df[col] = df[col].apply(lambda x: x[0] if len(x) == 1 else x)
    reshaped_data = []
    for col in df.columns:
        if 'Mean' in col:
            cell_name = col.split('(')[1].split(')')[0]
            for idx, value in enumerate(df[col]):
                if isinstance(value, list):
                    for sub_idx, sub_value in enumerate(value):
                        reshaped_data.append({
                            'Experiment': filename,
                            'Cell': cell_name,
                            'Timepoint': idx + sub_idx,
                            'Signal': sub_value
                        })
                else:
                    reshaped_data.append({
                        'Experiment': filename,
                        'Cell': cell_name,
                        'Timepoint': idx,
                        'Signal': value
                    })
    return pd.DataFrame(reshaped_data)

def subtract_background(non_bg_data, bg_data):
    result = []
    for exp in non_bg_data['Experiment'].unique():
        exp_data = non_bg_data[non_bg_data['Experiment'] == exp].copy()
        exp_bg = bg_data[bg_data['Experiment'] == exp]
        if not exp_bg.empty:
            bg_mean = exp_bg.groupby('Timepoint')['Signal'].mean()
            exp_data['Signal'] = exp_data.apply(lambda row: row['Signal'] - bg_mean[row['Timepoint']], axis=1)
        result.append(exp_data)
    return pd.concat(result, ignore_index=True)

def zscore_normalize(data, categorical_columns):
    numeric = data.drop(columns=categorical_columns)
    temporal = np.vstack(numeric.values)
    zscored = TimeSeriesScalerMeanVariance().fit_transform(temporal)[..., 0]
    df_zscored = pd.DataFrame(zscored, columns=numeric.columns)
    return pd.concat([data[categorical_columns].reset_index(drop=True),
                      df_zscored.reset_index(drop=True)], axis=1)

def decompose_data_rows(df, model='additive', freq=50):
    trends = []
    for idx, row in df.iterrows():
        signal = row.iloc[4:].dropna().values
        signal = signal[signal != 0]
        if len(signal) >= 2 * freq:
            decomp = seasonal_decompose(signal, model=model, period=freq)
            trend = decomp.trend[~np.isnan(decomp.trend)]
            padded = list(trend) + [0] * (len(row.iloc[4:]) - len(trend))
            trends.append(padded)
        else:
            trends.append([np.nan] * len(row.iloc[4:]))
    df_trend = pd.DataFrame(trends, columns=df.columns[4:])
    return pd.concat([df.iloc[:, :4].reset_index(drop=True), df_trend.reset_index(drop=True)], axis=1)

def calculate_auc(signal, timepoints, start_time, end_time):
    mask = (timepoints >= start_time) & (timepoints <= end_time)
    auc = np.trapz(signal[mask], timepoints[mask])
    return auc
