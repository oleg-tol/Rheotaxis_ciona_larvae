"""

Main script to run the brain imaging analysis pipeline.
It loads the data, performs preprocessing, PCA, custom analyses, and generates visualizations.
"""

import os
from datetime import date
import pandas as pd
from data_loading import load_data
from pca_analysis import perform_pca_on_entire_dataset
from brain_analysis import prepare_custom_datasets, print_cell_type_counts, generate_and_save_summary_data

directory = "path"
output_dir = os.path.join(directory, "Output_new")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
merged_df = load_data(directory)

today = date.today()
date_tag = today.strftime("%d%b%Y")
dest_folder = os.path.join('./results', date_tag)
os.makedirs(dest_folder, exist_ok=True)

exposure_time = 0.673474

from custom_analysis import prepare_custom_datasets
subsets = prepare_custom_datasets(merged_df)

from custom_analysis import print_cell_type_counts
print_cell_type_counts(subsets)

summary_data = generate_and_save_summary_data(subsets, dest_folder, exposure_time)
summary_data_df = pd.DataFrame.from_dict({(i, j): summary_data[i][j] 
                                          for i in summary_data.keys() 
                                          for j in summary_data[i].keys()}, orient='index').round(3)
summary_data_df.to_csv(os.path.join(dest_folder, "summary_data.csv"), index=False)
print(summary_data_df)

from pca_analysis import perform_pca_on_entire_dataset
pca_result, explained_variance = perform_pca_on_entire_dataset(merged_df, n_components=10)

