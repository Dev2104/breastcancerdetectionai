"""
4_Model_Insights.py

Dynamic model insights page with support for ALL trained models.
"""

from pathlib import Path
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# =========================
# PROJECT ROOT SETUP
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =========================
# PATHS
# =========================
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


# =========================
# LOAD FUNCTIONS
# =========================
def load_model_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_selected_model_bundle(model_name: str) -> dict:
    safe_name = model_name.lower().replace(" ", "_")
    model_path = MODELS_DIR / f"{safe_name}.pkl"
    return load_model_bundle(model_path)


# =========================
# FEATURE IMPORTANCE
# =========================
def get_feature_importance(model_bundle: dict) -> pd.DataFrame | None:
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        values = coef[0] if len(coef.shape) > 1 else coef
    else:
        return None

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": pd.Series(values).astype(float).abs()
    }).sort_values(by="importance", ascending=False)

    return df


# =========================
# MODEL DESCRIPTIONS
# =========================
def get_model_info(model_name: str) -> dict:
    return {
        "Logistic Regression": {
            "desc": "Linear model producing probability outputs.",
            "strengths": ["Interpretable", "Fast", "Reliable"],
            "weaknesses": ["Linear only", "Limited complexity"]
        },
        "Decision Tree": {
            "desc": "Rule-based hierarchical model.",
            "strengths": ["Easy to understand", "No scaling needed"],
            "weaknesses": ["Overfitting risk"]
        },
        "Random Forest": {
            "desc": "Ensemble of decision trees.",
            "strengths": ["High accuracy", "Stable"],
            "weaknesses": ["Less interpretable"]
        },
        "Support Vector Machine": {
            "desc": "Finds optimal separating boundary.",
            "strengths": ["Works well in complex spaces"],
            "weaknesses": ["Hard to interpret", "Needs scaling"]
        },
        "K-Nearest Neighbors": {
            "desc": "Distance-based classifier.",
            "strengths": ["Simple", "No training"],
            "weaknesses": ["Slow", "Sensitive to noise"]
        },
    }.get(model_name, {})


# =========================
# RENDER FUNCTIONS
# =========================
def render_feature_importance(df: pd.DataFrame):
    st.subheader("Feature Importance")

    top_df = df.head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_df, x="importance", y="feature", ax=ax)
    ax.set_title("Top 10 Features")

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(top_df, use_container_width=True)

    # Highlight top feature
    st.info(f"Top influencing feature: **{top_df.iloc[0]['feature']}**")


def render_interpretation(df: pd.DataFrame):
    st.subheader("Interpretation")

    st.write(
        "The model prioritizes structural and geometric tumor characteristics. "
        "These features help differentiate benign and malignant patterns."
    )

    st.success("Higher importance = stronger influence on prediction.")


# =========================
# MAIN
# =========================
def main():
    st.title("Model Insights")
    st.divider()

    try:
        best_bundle = load_model_bundle(BEST_MODEL_PATH)

        # =========================
        # BEST MODEL SUMMARY
        # =========================
        st.subheader("Current Best Model")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", best_bundle["model_name"])
        col2.metric("Scaling", "Yes" if best_bundle["scaling_required"] else "No")
        col3.metric("Features", len(best_bundle["feature_names"]))

        st.divider()
        st.subheader("Model Summary")
        st.success(f"Using **{best_bundle['model_name']}** as best-performing model.")

        # =========================
        # DROPDOWN
        # =========================
        st.divider()
        st.subheader("Explore Models")

        model_options = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "K-Nearest Neighbors",
        ]

        selected_model = st.selectbox(
            "Select a model:",
            model_options
        )

        if selected_model == best_bundle["model_name"]:
            st.success("This is the best-performing model.")

        # =========================
        # LOAD SELECTED MODEL
        # =========================
        selected_bundle = load_selected_model_bundle(selected_model)

        # =========================
        # MODEL INFO
        # =========================
        info = get_model_info(selected_model)

        st.subheader(f"{selected_model} Overview")
        st.info(info.get("desc", ""))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Strengths")
            for s in info.get("strengths", []):
                st.success(f"✔ {s}")

        with col2:
            st.subheader("Limitations")
            for w in info.get("weaknesses", []):
                st.error(f"✖ {w}")

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.divider()

        importance_df = get_feature_importance(selected_bundle)

        if importance_df is None or importance_df.empty:
            st.warning(
                "Feature importance not available for this model "
                "(common for SVM and KNN)."
            )
        else:
            render_feature_importance(importance_df)
            render_interpretation(importance_df)

    except Exception as e:
        st.error(f"Error: {e}")
        return

    # =========================
    # EXPLAINABILITY NOTE
    # =========================
    st.divider()
    st.subheader("Explainability Note")

    st.info(
        "Feature importance shows how much each input contributes to predictions. "
        "This improves transparency of the AI system."
    )

    # =========================
    # MODEL SELECTION
    # =========================
    st.divider()
    st.subheader("Model Selection")

    st.write(
        "The best model is selected based on accuracy and performance metrics "
        "such as precision, recall, and F1-score."
    )

    # =========================
    # DISCLAIMER
    # =========================
    st.divider()
    st.subheader("Disclaimer")

    st.warning(
        "This system is for research purposes only and not a medical diagnostic tool."
    )

    # =========================
    # FOOTER
    # =========================
    st.caption("Breast Cancer Detection AI • Model Insights • Research Use Only")


if __name__ == "__main__":
    main()