"""
3_Visualizations.py

Visualizations page for the Breast Cancer Detection AI Streamlit application.

This page provides visual analysis of the dataset and currently available
model outputs using a single dropdown-based interface.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure project root is available for imports inside the multipage app.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data  # noqa: E402
from src.explain import get_feature_importance  # noqa: E402
from src.unsupervised import run_pca_analysis  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"


def load_dataset() -> pd.DataFrame:
    """
    Load the breast cancer dataset using the existing project data loader.

    Returns
    -------
    pd.DataFrame
        Dataset containing feature columns, target, and target_label.
    """
    return load_data()


def load_model_comparison() -> pd.DataFrame | None:
    """
    Load model comparison results if the report exists.

    Returns
    -------
    pd.DataFrame | None
        Model comparison DataFrame if available, otherwise None.
    """
    if MODEL_COMPARISON_PATH.exists():
        return pd.read_csv(MODEL_COMPARISON_PATH)
    return None


def render_saved_figure(image_path: Path, title: str, missing_message: str) -> None:
    """
    Display a saved figure image if it exists, otherwise show an info message.

    Parameters
    ----------
    image_path : Path
        Path to the saved image file.
    title : str
        Section title to display.
    missing_message : str
        Message shown if the file does not exist.

    Returns
    -------
    None
    """
    st.subheader(title)

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(missing_message)


def render_class_distribution(df: pd.DataFrame) -> None:
    """
    Render class distribution chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the target_label column.

    Returns
    -------
    None
    """
    st.subheader("Class Distribution")
    st.write(
        "This chart shows how the dataset is distributed between benign and malignant cases."
    )

    class_counts = df["target_label"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        ax=ax
    )
    ax.set_xlabel("Tumor Class")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Benign and Malignant Cases")

    for index, value in enumerate(class_counts.values):
        ax.text(index, value, str(value), ha="center", va="bottom")

    st.pyplot(fig)
    plt.close(fig)


def render_feature_histograms(df: pd.DataFrame) -> None:
    """
    Render histograms for a selected feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns and target_label.

    Returns
    -------
    None
    """
    st.subheader("Feature Histograms")
    st.write(
        "Select a feature to view its distribution across the dataset."
    )

    excluded_columns = {"target", "target_label"}
    feature_columns = [col for col in df.columns if col not in excluded_columns]

    selected_feature = st.selectbox(
        "Select a feature",
        options=feature_columns,
        index=0,
        key="feature_histogram_selector"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=df,
        x=selected_feature,
        hue="target_label",
        kde=True,
        bins=30,
        ax=ax
    )
    ax.set_title(f"Histogram of {selected_feature}")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    plt.close(fig)


def render_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Render a correlation heatmap for the numeric feature set.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns.

    Returns
    -------
    None
    """
    st.subheader("Correlation Heatmap")
    st.write(
        "This heatmap shows correlations among the diagnostic input features."
    )

    numeric_df = df.drop(columns=["target", "target_label"]).copy()
    correlation_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")

    st.pyplot(fig)
    plt.close(fig)


def render_model_comparison() -> None:
    """
    Render model comparison chart using reports/model_comparison.csv if available.

    Returns
    -------
    None
    """
    st.subheader("Model Comparison")
    st.write(
        "This chart compares trained supervised learning models using saved report data."
    )

    comparison_df = load_model_comparison()

    if comparison_df is None:
        st.info(
            "Model comparison results are not available yet. "
            "Please run the training pipeline first to generate reports/model_comparison.csv."
        )
        return

    required_columns = {"Model", "Accuracy"}
    if not required_columns.issubset(comparison_df.columns):
        st.warning(
            "The model comparison report exists, but it does not contain the expected columns."
        )
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=comparison_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=20)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f")

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(comparison_df, use_container_width=True)


def render_pca_visualization() -> None:
    """
    Render PCA scatter plot using the unsupervised analysis module.

    Returns
    -------
    None
    """
    st.subheader("PCA Visualization")
    st.write(
        "This plot shows a 2D projection of the dataset using Principal Component Analysis (PCA). "
        "It helps visualize how well benign and malignant cases are separated."
    )

    try:
        pca_df = run_pca_analysis()
    except Exception as exc:
        st.error(f"Error running PCA: {exc}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="target_label",
        ax=ax
    )

    ax.set_title("PCA Projection (2D)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(pca_df.head(), use_container_width=True)


def render_feature_importance() -> None:
    """
    Render feature importance chart for the current best model.

    Returns
    -------
    None
    """
    st.subheader("Feature Importance")
    st.write(
        "This chart shows the most influential diagnostic features used by the "
        "current best-performing model."
    )

    try:
        importance_df = get_feature_importance()
    except ValueError as exc:
        st.info(str(exc))
        return
    except Exception as exc:
        st.error(f"Unable to load feature importance: {exc}")
        return

    top_n = st.slider(
        "Select number of top features to display",
        min_value=5,
        max_value=min(20, len(importance_df)),
        value=min(10, len(importance_df)),
        step=1,
        key="feature_importance_top_n",
    )

    top_df = importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top_df,
        x="Absolute Importance",
        y="Feature",
        ax=ax
    )
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Absolute Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(top_df, use_container_width=True)


def main() -> None:
    """
    Main entry point for the Visualizations page.

    Returns
    -------
    None
    """
    st.title("Visualizations")
    st.write(
        "This page provides visual analysis of the breast cancer dataset and "
        "currently available model outputs to support interpretation and thesis presentation."
    )

    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")
        return

    visualization_option = st.selectbox(
        "Select a visualization",
        options=[
            "Class Distribution",
            "Feature Histograms",
            "Correlation Heatmap",
            "Model Comparison",
            "Confusion Matrix",
            "ROC Curve",
            "PCA Visualization",
            "Feature Importance",
        ],
        index=0
    )

    if visualization_option == "Class Distribution":
        render_class_distribution(df)
    elif visualization_option == "Feature Histograms":
        render_feature_histograms(df)
    elif visualization_option == "Correlation Heatmap":
        render_correlation_heatmap(df)
    elif visualization_option == "Model Comparison":
        render_model_comparison()
    elif visualization_option == "Confusion Matrix":
        render_saved_figure(
            image_path=CONFUSION_MATRIX_PATH,
            title="Confusion Matrix",
            missing_message=(
                "Confusion matrix image is not available yet. "
                "Please run the training pipeline first to generate "
                "reports/figures/confusion_matrix.png."
            ),
        )
    elif visualization_option == "ROC Curve":
        render_saved_figure(
            image_path=ROC_CURVE_PATH,
            title="ROC Curve",
            missing_message=(
                "ROC curve image is not available yet. "
                "Please run the training pipeline first to generate "
                "reports/figures/roc_curve.png."
            ),
        )
    elif visualization_option == "PCA Visualization":
        render_pca_visualization()
    elif visualization_option == "Feature Importance":
        render_feature_importance()


if __name__ == "__main__":
    main()