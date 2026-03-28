"""
data_loader.py

This module is responsible for loading the Breast Cancer Wisconsin dataset
from sklearn, converting it into a pandas DataFrame, and providing reusable
functions to access features (X) and target (y).

The dataset is labeled as:
- 0 -> malignant
- 1 -> benign

For clarity, we also include a human-readable label column.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data(as_dataframe: bool = True) -> pd.DataFrame:
    """
    Load the Breast Cancer Wisconsin dataset and return it as a pandas DataFrame.

    Parameters
    ----------
    as_dataframe : bool, optional (default=True)
        If True, returns the dataset as a pandas DataFrame.
        If False, raises ValueError (only DataFrame supported for this project).

    Returns
    -------
    pd.DataFrame
        DataFrame containing features and target columns.
    """
    if not as_dataframe:
        raise ValueError("Only DataFrame format is supported in this project.")

    # Load dataset from sklearn
    data = load_breast_cancer()

    # Create DataFrame for features
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add target column (numeric)
    df["target"] = data.target

    # Add human-readable target labels
    # Note: sklearn uses 0 = malignant, 1 = benign
    df["target_label"] = df["target"].map({0: "malignant", 1: "benign"})

    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split the dataset into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset DataFrame returned by load_data().

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except target columns).

    y : pd.Series
        Target vector (0 = malignant, 1 = benign).
    """
    if "target" not in df.columns:
        raise ValueError("DataFrame must contain 'target' column.")

    # Drop target and label columns to get features
    X = df.drop(columns=["target", "target_label"])

    # Target column
    y = df["target"]

    return X, y


def get_feature_names() -> list:
    """
    Get the list of feature names from the dataset.

    Returns
    -------
    list
        List of feature names.
    """
    data = load_breast_cancer()
    return list(data.feature_names)


def get_target_names() -> dict:
    """
    Get mapping of target values to class labels.

    Returns
    -------
    dict
        Mapping of target integer to class label.
    """
    return {
        0: "malignant",
        1: "benign"
    }


if __name__ == "__main__":
    # Quick test to ensure module works independently
    df = load_data()
    X, y = get_features_and_target(df)

    print("Dataset shape:", df.shape)
    print("Features shape:", X.shape)
    print("Target distribution:\n", y.value_counts())
    print("Sample data:\n", df.head())