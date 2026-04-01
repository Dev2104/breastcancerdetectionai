"""
unsupervised.py

This module provides unsupervised exploratory analysis utilities for the
Breast Cancer Detection AI project.

Currently supported:
- PCA dimensionality reduction for 2D visualization

The module is designed to work with the existing project structure and
current data loading logic without modifying the supervised pipeline.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import get_features_and_target, load_data


def run_pca_analysis() -> pd.DataFrame:
    """
    Run PCA analysis on the breast cancer dataset and return a 2D projection.

    The function:
    - loads the dataset using the current data loader
    - splits features and target using the existing project logic
    - scales the feature matrix using StandardScaler
    - applies PCA with 2 principal components
    - returns a DataFrame containing:
        - PC1
        - PC2
        - target
        - target_label

    Returns
    -------
    pd.DataFrame
        DataFrame containing the 2D PCA projection and target information.
    """
    df = load_data()
    X, y = get_features_and_target(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio: PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}")

    pca_df = pd.DataFrame(
        principal_components,
        columns=["PC1", "PC2"],
        index=df.index
    )

    pca_df["target"] = y.values
    pca_df["target_label"] = df["target_label"].values

    return pca_df


if __name__ == "__main__":
    pca_output = run_pca_analysis()
    print(pca_output.head())