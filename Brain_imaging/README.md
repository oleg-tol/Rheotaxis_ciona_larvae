# Brain Imaging Data Analysis

This repository contains a modular pipeline for calcium imaging data analysis. The project is divided into the following modules:

- **data_loading.py:**
  Functions to load and merge CSV files from a specified directory.

- **preprocessing.py:**
  Functions for preprocessing and z-score normalization of time-series data.

- **pca_analysis.py:**
  Functions to perform PCA on z-scored data and the entire dataset.

- **pca_visualization.py:**
  Functions to generate clustermaps, 3D PCA trajectory plots, and other visualizations.

- **brain_analysis.py:**
  Functions for custom analyses including phase splitting, cell grouping, and summary score computation.

- **main.py:**
  The main script that start the analysis pipeline.

## Requirements

- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, tslearn

## Setup

1. Install the required packages:
    
2. Update file paths and parameters as needed in the source files.

3. Run the analysis pipeline.  
