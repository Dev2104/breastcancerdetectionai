"""
train.py

Trains multiple ML models, selects the best one, and saves:
1. best_model.pkl (for app usage)
2. all individual model bundles (for insights/comparison)
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


# =========================
# PROJECT ROOT
# =========================
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# =========================
# MODELS
# =========================
def get_models() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


def get_scaled_model_names() -> set:
    return {
        "Logistic Regression",
        "Support Vector Machine",
        "K-Nearest Neighbors",
    }


# =========================
# TRAIN / EVALUATE
# =========================
def train_single_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model.fit(X_train, y_train)
    return model


def evaluate_single_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)


# =========================
# TRAIN ALL MODELS
# =========================
def train_and_compare_models(
    X_train_unscaled,
    X_test_unscaled,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
):
    models = get_models()
    scaled_model_names = get_scaled_model_names()

    trained_models = {}
    results = []

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

        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
        })

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]

    return trained_models, results_df, best_model_name


# =========================
# MODEL BUNDLE
# =========================
def create_model_bundle(
    model,
    model_name,
    scaler,
    feature_names,
):
    scaling_required = model_name in get_scaled_model_names()

    return {
        "model": model,
        "scaler": scaler if scaling_required else None,
        "model_name": model_name,
        "scaling_required": scaling_required,
        "feature_names": feature_names,
    }


def save_model_bundle(model_bundle: dict, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, file_path)


# =========================
# SAVE ALL MODELS (NEW)
# =========================
def save_all_models(
    trained_models: Dict[str, Any],
    scaler: StandardScaler,
    feature_names: List[str],
    project_root: Path,
):
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    for model_name, model in trained_models.items():
        safe_name = model_name.lower().replace(" ", "_")

        model_bundle = create_model_bundle(
            model=model,
            model_name=model_name,
            scaler=scaler,
            feature_names=feature_names,
        )

        model_path = models_dir / f"{safe_name}.pkl"
        joblib.dump(model_bundle, model_path)


# =========================
# PIPELINE
# =========================
def run_training_pipeline():
    df = load_data()
    X, y = get_features_and_target(df)
    feature_names = list(X.columns)

    X_train_unscaled, X_test_unscaled, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train=X_train_unscaled,
        X_test=X_test_unscaled,
    )

    trained_models, results_df, best_model_name = train_and_compare_models(
        X_train_unscaled,
        X_test_unscaled,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    best_model = trained_models[best_model_name]

    project_root = get_project_root()

    # ✅ Save BEST model (existing behavior)
    best_bundle = create_model_bundle(
        model=best_model,
        model_name=best_model_name,
        scaler=scaler,
        feature_names=feature_names,
    )

    best_model_path = project_root / "models" / "best_model.pkl"
    save_model_bundle(best_bundle, best_model_path)

    # 🔥 NEW: Save ALL models
    save_all_models(
        trained_models=trained_models,
        scaler=scaler,
        feature_names=feature_names,
        project_root=project_root,
    )

    return trained_models, results_df, best_model_name


# =========================
# MAIN
# =========================
def main():
    _, results_df, best_model_name = run_training_pipeline()

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_model_name}")
    print("All models saved in /models directory")


if __name__ == "__main__":
    main()