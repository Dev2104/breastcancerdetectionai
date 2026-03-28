"""
train.py

This module trains multiple supervised machine learning models for breast cancer
classification, compares them using accuracy, and saves the best-performing model
in a deployment-safe format.

Models included:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. K-Nearest Neighbors

Scaling strategy:
- Scaled data: Logistic Regression, Support Vector Machine, K-Nearest Neighbors
- Unscaled data: Decision Tree, Random Forest

The saved model file contains a bundle with:
- trained best model
- scaler (or None if not needed)
- best model name
- whether scaling is required
- feature names

This module is designed to run with:
    python -m src.train
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import get_features_and_target, load_data
from src.preprocess import scale_data, split_data


def get_project_root() -> Path:
    """
    Return the project root directory.

    Returns
    -------
    Path
        Path to the project root directory.
    """
    return Path(__file__).resolve().parent.parent


def get_models() -> Dict[str, Any]:
    """
    Create and return all machine learning models used in the project.

    Returns
    -------
    Dict[str, Any]
        Dictionary where keys are model names and values are model instances.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


def get_scaled_model_names() -> set:
    """
    Return the set of model names that require scaled input data.

    Returns
    -------
    set
        Set of model names that should be trained on scaled data.
    """
    return {
        "Logistic Regression",
        "Support Vector Machine",
        "K-Nearest Neighbors",
    }


def train_single_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a single machine learning model.

    Parameters
    ----------
    model : Any
        The machine learning model to train.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.

    Returns
    -------
    Any
        Trained model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_single_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluate a trained model using accuracy.

    Parameters
    ----------
    model : Any
        Trained machine learning model.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.

    Returns
    -------
    float
        Accuracy score.
    """
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


def train_and_compare_models(
    X_train_unscaled: pd.DataFrame,
    X_test_unscaled: pd.DataFrame,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """
    Train all models, compare them using accuracy, and identify the best model.

    Parameters
    ----------
    X_train_unscaled : pd.DataFrame
        Unscaled training feature matrix.
    X_test_unscaled : pd.DataFrame
        Unscaled testing feature matrix.
    X_train_scaled : pd.DataFrame
        Scaled training feature matrix.
    X_test_scaled : pd.DataFrame
        Scaled testing feature matrix.
    y_train : pd.Series
        Training target vector.
    y_test : pd.Series
        Testing target vector.

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame, str]
        - Dictionary of trained models
        - DataFrame with model comparison results
        - Name of the best model
    """
    models = get_models()
    scaled_model_names = get_scaled_model_names()

    trained_models: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    for model_name, model in models.items():
        if model_name in scaled_model_names:
            X_train = X_train_scaled
            X_test = X_test_scaled
        else:
            X_train = X_train_unscaled
            X_test = X_test_unscaled

        trained_model = train_single_model(model, X_train, y_train)
        accuracy = evaluate_single_model(trained_model, X_test, y_test)

        trained_models[model_name] = trained_model
        results.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy, 4),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]

    return trained_models, results_df, best_model_name


def create_model_bundle(
    best_model: Any,
    best_model_name: str,
    scaler: StandardScaler,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Create a deployment-safe bundle for the best model.

    Parameters
    ----------
    best_model : Any
        Trained best model.
    best_model_name : str
        Name of the best model.
    scaler : StandardScaler
        Fitted scaler from preprocessing.
    feature_names : List[str]
        List of feature names used during training.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing model, scaler, metadata, and feature names.
    """
    scaling_required = best_model_name in get_scaled_model_names()

    model_bundle = {
        "model": best_model,
        "scaler": scaler if scaling_required else None,
        "model_name": best_model_name,
        "scaling_required": scaling_required,
        "feature_names": feature_names,
    }

    return model_bundle


def save_best_model_bundle(model_bundle: Dict[str, Any], file_path: Path) -> None:
    """
    Save the deployment-safe model bundle using joblib.

    Parameters
    ----------
    model_bundle : Dict[str, Any]
        Dictionary containing the trained model and deployment metadata.
    file_path : Path
        File path where the bundle will be saved.

    Returns
    -------
    None
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, file_path)


def run_training_pipeline() -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """
    Run the full training pipeline:
    1. Load the dataset
    2. Split unscaled data
    3. Preprocess scaled data
    4. Train all models
    5. Compare model performance
    6. Save the best model as a deployment-safe bundle

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame, str]
        - Dictionary of trained models
        - DataFrame of model comparison results
        - Name of the best model
    """
    df = load_data()
    X, y = get_features_and_target(df)
    feature_names = list(X.columns)

            # Single split for all models
    X_train_unscaled, X_test_unscaled, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    # Scale the same split for scale-sensitive models
    from src.preprocess import scale_data
    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train=X_train_unscaled,
        X_test=X_test_unscaled,
    )

    trained_models, results_df, best_model_name = train_and_compare_models(
        X_train_unscaled=X_train_unscaled,
        X_test_unscaled=X_test_unscaled,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
    )

    best_model = trained_models[best_model_name]

    model_bundle = create_model_bundle(
        best_model=best_model,
        best_model_name=best_model_name,
        scaler=scaler,
        feature_names=feature_names,
    )

    project_root = get_project_root()
    best_model_path = project_root / "models" / "best_model.pkl"
    save_best_model_bundle(model_bundle, best_model_path)

    return trained_models, results_df, best_model_name


def main() -> None:
    """
    Main entry point for training and saving the best model.

    Returns
    -------
    None
    """
    _, results_df, best_model_name = run_training_pipeline()

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_model_name}")
    print("Deployment-safe best model bundle saved to: models/best_model.pkl")


if __name__ == "__main__":
    main()