"""
evaluate.py

Evaluation utilities for the breast cancer detection project.

This module provides reusable functions for:
- calculating classification metrics
- generating confusion matrix plots
- generating ROC curve plots
- performing cross-validation

Designed to work with the current project structure and training pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score


def evaluate_model(
    y_true: pd.Series,
    y_pred: pd.Series,
    model: Optional[Any] = None,
    X_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[float]]:
    """
    Evaluate a classification model using standard metrics.

    Parameters
    ----------
    y_true : pd.Series
        True target labels.
    y_pred : pd.Series
        Predicted target labels.
    model : Optional[Any], optional
        Trained model object. Used for ROC-AUC if it supports predict_proba.
    X_test : Optional[pd.DataFrame], optional
        Test feature matrix. Required for ROC-AUC calculation when model is provided.

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary containing:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC AUC
    """
    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC AUC": None,
    }

    if model is not None and X_test is not None and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["ROC AUC"] = round(roc_auc_score(y_true, y_prob), 4)

    return metrics


def generate_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : pd.Series
        True target labels.
    y_pred : pd.Series
        Predicted target labels.
    save_path : Path
        File path where the confusion matrix image will be saved.
    title : str, optional
        Plot title. Default is "Confusion Matrix".

    Returns
    -------
    None
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path,
    title: str = "ROC Curve",
) -> Optional[float]:
    """
    Generate and save an ROC curve plot if the model supports probability prediction.

    Parameters
    ----------
    model : Any
        Trained model object.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        True target labels for the test set.
    save_path : Path
        File path where the ROC curve image will be saved.
    title : str, optional
        Plot title. Default is "ROC Curve".

    Returns
    -------
    Optional[float]
        ROC-AUC score if generated successfully, otherwise None.
    """
    if not hasattr(model, "predict_proba"):
        return None

    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return round(roc_auc, 4)


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Dict[str, float]:
    """
    Perform cross-validation for a model.

    Parameters
    ----------
    model : Any
        Machine learning model to evaluate.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    cv : int, optional
        Number of folds. Default is 5.
    scoring : str, optional
        Scoring metric for cross-validation. Default is "accuracy".

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean and standard deviation of cross-validation scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return {
        "CV Mean": round(scores.mean(), 4),
        "CV Std": round(scores.std(), 4),
    }