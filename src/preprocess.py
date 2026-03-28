"""
preprocess.py

This module handles data preprocessing for the breast cancer detection project.

It provides reusable functions to:
1. split the dataset into training and testing sets
2. scale feature values using StandardScaler when needed

The module is designed to work directly with the outputs from src.data_loader.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into training and testing sets.
    """
    if X.empty:
        raise ValueError("Feature matrix X is empty.")

    if y.empty:
        raise ValueError("Target vector y is empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )

    return X_train, X_test, y_train, y_test


def scale_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale training and testing feature sets using StandardScaler.
    Fit scaler on training data only, then transform both train and test data.
    """
    if X_train.empty:
        raise ValueError("X_train is empty.")

    if X_test.empty:
        raise ValueError("X_test is empty.")

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    from src.data_loader import load_data, get_features_and_target

    df = load_data()
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("Preprocessing completed successfully.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Scaler mean (first 5 values):", scaler.mean_[:5])