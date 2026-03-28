"""
predict.py

This module handles inference for the breast cancer detection project.

It loads the deployment-safe model bundle saved by src.train and provides
reusable functions for:
1. loading the model bundle
2. preparing input data
3. making a single prediction
4. making batch predictions

The module automatically applies scaling when required and ensures that
input feature order matches the feature names used during training.

This module is designed to run with:
    python -m src.predict
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

from src.data_loader import get_features_and_target, get_target_names, load_data


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns
    -------
    Path
        Path to the project root directory.
    """
    return Path(__file__).resolve().parent.parent


def get_model_path() -> Path:
    """
    Return the path to the saved model bundle.

    Returns
    -------
    Path
        Path to models/best_model.pkl
    """
    return get_project_root() / "models" / "best_model.pkl"


def load_model_bundle(model_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load the deployment-safe model bundle from disk.

    Parameters
    ----------
    model_path : Optional[Union[str, Path]], optional
        Custom path to the saved model bundle. If None, the default
        models/best_model.pkl path is used.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - model
        - scaler
        - model_name
        - scaling_required
        - feature_names

    Raises
    ------
    FileNotFoundError
        If the model bundle file does not exist.
    ValueError
        If the loaded bundle is missing required keys.
    """
    resolved_path = Path(model_path) if model_path is not None else get_model_path()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at: {resolved_path}. "
            "Please run training first using: python -m src.train"
        )

    model_bundle = joblib.load(resolved_path)

    required_keys = {
        "model",
        "scaler",
        "model_name",
        "scaling_required",
        "feature_names",
    }

    missing_keys = required_keys.difference(model_bundle.keys())
    if missing_keys:
        raise ValueError(
            f"Loaded model bundle is missing required keys: {sorted(missing_keys)}"
        )

    return model_bundle


def prepare_input_data(
    input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    feature_names: Optional[list] = None
) -> pd.DataFrame:
    """
    Prepare input data for prediction and ensure correct feature order.

    Parameters
    ----------
    input_data : Union[Dict[str, Any], pd.Series, pd.DataFrame]
        Input feature data for prediction.
        Supported formats:
        - dictionary for a single sample
        - pandas Series for a single sample
        - pandas DataFrame for one or more samples
    feature_names : Optional[list], optional
        Ordered list of expected feature names. If None, feature names
        are loaded from the saved model bundle.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with columns ordered exactly as required.

    Raises
    ------
    TypeError
        If input_data is not a supported type.
    ValueError
        If required features are missing or data contains invalid columns.
    """
    if feature_names is None:
        model_bundle = load_model_bundle()
        feature_names = model_bundle["feature_names"]

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError(
            "input_data must be a dictionary, pandas Series, or pandas DataFrame."
        )

    missing_features = [feature for feature in feature_names if feature not in input_df.columns]
    if missing_features:
        raise ValueError(
            f"Input data is missing required features: {missing_features}"
        )

    input_df = input_df[feature_names].copy()

    for column in feature_names:
        input_df[column] = pd.to_numeric(input_df[column], errors="raise")

    return input_df


def get_prediction_label(prediction: int) -> str:
    """
    Convert numeric prediction into human-readable class label.

    Parameters
    ----------
    prediction : int
        Numeric class prediction.

    Returns
    -------
    str
        Human-readable label: malignant or benign.
    """
    target_names = get_target_names()
    return target_names.get(int(prediction), "unknown")


def predict_single(input_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
    """
    Predict the class for a single input sample.

    Parameters
    ----------
    input_data : Union[Dict[str, Any], pd.Series, pd.DataFrame]
        Single input sample for prediction.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - model_name
        - scaling_required
        - prediction
        - prediction_label
        - probabilities (if supported)
    """
    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    model_name = model_bundle["model_name"]
    scaling_required = model_bundle["scaling_required"]
    feature_names = model_bundle["feature_names"]

    prepared_input = prepare_input_data(input_data, feature_names=feature_names)

    if len(prepared_input) != 1:
        raise ValueError("predict_single expects exactly one input sample.")

    if scaling_required and scaler is not None:
        processed_input = scaler.transform(prepared_input)
    else:
        processed_input = prepared_input

    prediction = int(model.predict(processed_input)[0])
    prediction_label = get_prediction_label(prediction)

    result = {
        "model_name": model_name,
        "scaling_required": scaling_required,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "probabilities": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_input)[0]
        result["probabilities"] = {
            "malignant": float(probabilities[0]),
            "benign": float(probabilities[1]),
        }

    return result


def predict_batch(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict classes for multiple input samples.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing one or more input samples.

    Returns
    -------
    pd.DataFrame
        DataFrame containing original ordered features plus:
        - prediction
        - prediction_label
        - probability_malignant (if supported)
        - probability_benign (if supported)
    """
    model_bundle = load_model_bundle()
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    scaling_required = model_bundle["scaling_required"]
    feature_names = model_bundle["feature_names"]

    prepared_input = prepare_input_data(input_df, feature_names=feature_names)

    if scaling_required and scaler is not None:
        processed_input = scaler.transform(prepared_input)
    else:
        processed_input = prepared_input

    predictions = model.predict(processed_input)

    results_df = prepared_input.copy()
    results_df["prediction"] = predictions.astype(int)
    results_df["prediction_label"] = results_df["prediction"].apply(get_prediction_label)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_input)
        results_df["probability_malignant"] = probabilities[:, 0].astype(float)
        results_df["probability_benign"] = probabilities[:, 1].astype(float)

    return results_df


def main() -> None:
    """
    Run a simple demo prediction using one sample row from the dataset.

    Returns
    -------
    None
    """
    df = load_data()
    X, _ = get_features_and_target(df)

    sample_row = X.iloc[0].to_dict()
    prediction_result = predict_single(sample_row)

    print("\nDemo Prediction")
    print("-" * 50)
    print(f"Model Name         : {prediction_result['model_name']}")
    print(f"Scaling Required   : {prediction_result['scaling_required']}")
    print(f"Numeric Prediction : {prediction_result['prediction']}")
    print(f"Predicted Label    : {prediction_result['prediction_label']}")

    if prediction_result["probabilities"] is not None:
        print("Prediction Probabilities:")
        print(f"  Malignant: {prediction_result['probabilities']['malignant']:.4f}")
        print(f"  Benign   : {prediction_result['probabilities']['benign']:.4f}")


if __name__ == "__main__":
    main()