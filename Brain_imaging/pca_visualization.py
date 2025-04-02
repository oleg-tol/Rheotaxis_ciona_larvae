"""
Functions for generating visualizations.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

def generate_zscore_clustermaps(df_zscored, sample_id, savepath):
   try:
        scale_cg1 = (len(df_zscored.columns)) / 2
        fig = sns.clustermap(data=df_zscored.T, col_cluster=False, figsize=(15, scale_cg1))
        save_dir = os.path.join(savepath, 'clustermaps')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'clustermap_zscored_{sample_id}.svg'))
    except Exception as e:
        print(f"Error generating clustermap: {e}")

def generate_pca_loadings_clustermap(df_loadings, sample_id, savepath):
   try:
        scale_cg1 = (len(df_loadings.columns)) / 2
        fig = sns.clustermap(df_loadings.T, cmap="PiYG", col_cluster=False, figsize=(3, scale_cg1))
        save_dir = os.path.join(savepath, 'loadings_clustermaps')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'clustermap_pca_loadings_{sample_id}.svg'))
    except Exception as e:
        print(f"Error generating PCA loadings clustermap: {e}")

def plot_pca_traj3d(results_pca, sample_id, savepath, aten_max_times=[], palp_max_times=[], exposure_time=0.673474):
    if results_pca is None or results_pca.shape[1] < 3:
        print(f"Skipping 3D plot for {sample_id} due to insufficient PCA components.")
        return
    try:
        N = results_pca.shape[0]
        T = results_pca
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect('equal')
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0, vmax=N)
        
        for i in range(1, N):
            ax.plot(T[i-1:i+1, 2], T[i-1:i+1, 0], T[i-1:i+1, 1], color=cmap(norm(i)))
            if i == int(61 / exposure_time):
                ax.scatter(T[i, 2], T[i, 0], T[i, 1], color='green', s=50, marker='o', label='Start Stimuli')
            elif i == int((61 + 30) / exposure_time):
                ax.scatter(T[i, 2], T[i, 0], T[i, 1], color='red', s=50, marker='x', label='End Stimuli')
        
        def plot_max_transients(transient_times, color, label):
            for time in transient_times:
                time_idx = int(float(time))
                if 2 <= time_idx < len(T):
                    ax.scatter(T[time_idx, 2], T[time_idx, 0], T[time_idx, 1], color=color, s=100, marker='*', label=label if time==transient_times[0] else "")
        
        plot_max_transients(aten_max_times, 'orange', 'Aten max transient')
        plot_max_transients(palp_max_times, 'cyan', 'Palp max transient')
        
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())
        
        ax.set_xlim(T[:, 2].min()-2, T[:, 2].max()+2)
        ax.set_ylim(T[:, 0].min()-2, T[:, 0].max()+2)
        ax.set_zlim(T[:, 1].min()-2, T[:, 1].max()+2)
        ax.view_init(elev=30., azim=40.)
        
        save_dir = os.path.join(savepath, 'pca3dplots_with_stims')
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'plot3d_{sample_id}.svg'))
        plt.show()
    except Exception as e:
        print(f"Error plotting 3D trajectory: {e}")
