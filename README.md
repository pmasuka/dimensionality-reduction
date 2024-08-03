# Cluster Countries Based on Happiness Indicators

## Overview

This project aims to cluster countries based on various indicators of happiness using several dimensionality reduction techniques, including PCA, LDA, t-SNE, and UMAP. Using Bokeh to provide interactive plots, the data is visualized, with country names and happiness scores displayed.

## Installation

To run this project, you need to install the following Python packages:

- umap
- umap-learn
- pandas
- matplotlib
- datashader
- bokeh
- holoviews
- scikit-image
- colorcet
- numpy
- scikit-learn

You can install these packages using pip:

```bash
pip install umap umap-learn pandas matplotlib datashader bokeh holoviews scikit-image colorcet numpy scikit-learn
```

## Usage

1. **Upload the Dataset**: Upload your dataset file (CSV format) to the environment.

2. **Read the Dataset**: Read the uploaded CSV file into a pandas DataFrame.

3. **Preprocess Data**: Preprocess the data by selecting numerical attributes and applying one-hot encoding to categorical data.

4. **Dimensionality Reduction**:
    - **PCA**: Perform Principal Component Analysis and visualize the data in 2D.
    - **LDA**: Use Linear Discriminant Analysis to maximize between-category distances.
    - **t-SNE**: Apply t-SNE for nonlinear dimensionality reduction.
    - **UMAP**: Experiment with different hyperparameters to observe their influence on the visualization.

5. **Visualization**: Visualize the reduced data using Bokeh, with interactive plots displaying country names and happiness scores.

## Dimensionality Reduction Techniques

### Principal Component Analysis (PCA)

PCA reduces the dimensionality of the data while preserving as much variance as possible. The first two principal components explain 65% of the variance, allowing for a 2D visualization.

### Linear Discriminant Analysis (LDA)

LDA is a supervised technique that maximizes the separation between categories (e.g., happiness levels). It provides components that can be visualized in 2D.

### t-SNE

t-SNE is a nonlinear dimensionality reduction technique that excels at preserving local structures in the data. It provides a 2D representation of high-dimensional data.

### UMAP

UMAP is a versatile technique for dimensionality reduction that can be tuned using parameters such as the number of neighbors, minimum distance, and metric. The project experiments with these parameters to observe their impact on the resulting visualization.

## Hyperparameter Tuning for UMAP

### Number of Neighbors

Controls the balance between local and global structure in the data. The default value is 15. The project explores values of 2, 15, and 30.

### Minimum Distance

Determines the minimum distance between points in the low-dimensional space. The default value is 0.1. The project explores values of 0.0, 0.1, and 0.99.

### Metric

Defines how distances are computed in the input data space. The project uses Euclidean, Mahalanobis, and Cosine metrics for comparison.

## Results and Observations

- **PCA**: Most variance is explained by the first two components. Clear distinction between countries with high and low happiness scores.
- **LDA**: Effective at separating countries into distinct happiness categories.
- **t-SNE**: Provides a clear representation of clusters, though interpretation requires careful parameter tuning.
- **UMAP**: Shows apparent clustering with different hyperparameters affecting the granularity and separation of clusters.

## Conclusion

The project demonstrates the effectiveness of various dimensionality reduction techniques in clustering countries based on happiness indicators. Visualization using Bokeh provides an interactive way to explore the data and gain insights into the factors contributing to happiness across different countries.
