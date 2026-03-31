"""
train.py

Full training pipeline for Breast Cancer Detection AI.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import get_features_and_target, load_data
from src.evaluate import generate_confusion_matrix, generate_roc_curve
from src.preprocess import scale_data, split_data


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


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


def train_single_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model.fit(X_train, y_train)
    return model


def evaluate_single_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4) if roc_auc else None,
    }


def save_model_comparison_report(results_df: pd.DataFrame, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(file_path, index=False)


def train_and_compare_models(
    X_train_unscaled,
    X_test_unscaled,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:

    models = get_models()
    scaled_model_names = get_scaled_model_names()

    trained_models = {}
    results = []

    for model_name, model in models.items():
        if model_name in scaled_model_names:
            X_train, X_test = X_train_scaled, X_test_scaled
        else:
            X_train, X_test = X_train_unscaled, X_test_unscaled

        trained_model = train_single_model(model, X_train, y_train)
        metrics = evaluate_single_model(trained_model, X_test, y_test)

        trained_models[model_name] = trained_model
        results.append({"Model": model_name, **metrics})

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]

    return trained_models, results_df, best_model_name


def create_model_bundle(
    best_model,
    best_model_name,
    scaler,
    feature_names,
):
    scaling_required = best_model_name in get_scaled_model_names()

    return {
        "model": best_model,
        "scaler": scaler if scaling_required else None,
        "model_name": best_model_name,
        "scaling_required": scaling_required,
        "feature_names": feature_names,
    }


def save_best_model_bundle(model_bundle, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, file_path)


def run_training_pipeline():

    df = load_data()
    X, y = get_features_and_target(df)
    feature_names = list(X.columns)

    X_train_unscaled, X_test_unscaled, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(
        X_train_unscaled, X_test_unscaled
    )

    trained_models, results_df, best_model_name = train_and_compare_models(
        X_train_unscaled,
        X_test_unscaled,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    project_root = get_project_root()

    # Save CSV
    reports_dir = project_root / "reports"
    model_comparison_path = reports_dir / "model_comparison.csv"
    save_model_comparison_report(results_df, model_comparison_path)

    # Best model
    best_model = trained_models[best_model_name]

    if best_model_name in get_scaled_model_names():
        X_test_for_eval = X_test_scaled
    else:
        X_test_for_eval = X_test_unscaled

    y_pred = best_model.predict(X_test_for_eval)

    # 🔥 FIXED PART (IMPORTANT)
    figures_dir = reports_dir / "figures"

    # Confusion Matrix
    confusion_matrix_path = figures_dir / "confusion_matrix.png"
    generate_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        save_path=confusion_matrix_path,
    )

    # ROC Curve
    roc_curve_path = figures_dir / "roc_curve.png"
    generate_roc_curve(
        model=best_model,
        X_test=X_test_for_eval,
        y_test=y_test,
        save_path=roc_curve_path,
    )

    # Save model
    model_bundle = create_model_bundle(
        best_model,
        best_model_name,
        scaler,
        feature_names,
    )

    best_model_path = project_root / "models" / "best_model.pkl"
    save_best_model_bundle(model_bundle, best_model_path)

    return trained_models, results_df, best_model_name


def main():
    _, results_df, best_model_name = run_training_pipeline()

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_model_name}")
    print("Saved: models/best_model.pkl")
    print("Saved: reports/model_comparison.csv")
    print("Saved: reports/figures/confusion_matrix.png")
    print("Saved: reports/figures/roc_curve.png")


if __name__ == "__main__":
    main()