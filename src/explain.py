"""
explain.py

Explainability utilities for the Breast Cancer Detection AI project.

This module extracts feature importance / interpretability information
from the saved deployment-safe model bundle.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.predict import load_model_bundle


def get_feature_importance() -> pd.DataFrame:
    """
    Extract feature importance information from the saved model bundle.

    Supports:
    - Logistic Regression via model coefficients
    - Random Forest via feature_importances_

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        - Feature
        - Coefficient
        - Absolute Importance

    Raises
    ------
    ValueError
        If the current best model does not support feature importance extraction.
    """
    model_bundle = load_model_bundle()

    model: Any = model_bundle["model"]
    model_name: str = model_bundle["model_name"]
    feature_names: list[str] = model_bundle["feature_names"]

    if model_name == "Logistic Regression":
        coefficients = model.coef_[0]

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": coefficients,
            }
        )
        importance_df["Absolute Importance"] = importance_df["Coefficient"].abs()

        return importance_df.sort_values(
            by="Absolute Importance",
            ascending=False
        ).reset_index(drop=True)

    if model_name == "Random Forest":
        importances = model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": importances,
            }
        )
        importance_df["Absolute Importance"] = importance_df["Coefficient"].abs()

        return importance_df.sort_values(
            by="Absolute Importance",
            ascending=False
        ).reset_index(drop=True)

    raise ValueError(
        f"Feature importance is not currently supported for the best model: {model_name}"
    )


if __name__ == "__main__":
    feature_importance_df = get_feature_importance()
    print(feature_importance_df.head(10))