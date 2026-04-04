from pathlib import Path
import sys
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# PROJECT ROOT SETUP
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Glass UI Components
from UI.ui_master import (
    configure_page,
    inject_master_theme,
    render_page_header,
    render_section_title,
    render_metric_card,
    render_divider,
    render_footer_note,
    render_card
)

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Setup Page & Theme
configure_page("Model Insights | Breast Cancer AI")
inject_master_theme()

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
    # FIXED: Replaced the incorrect ')' with '}' at the end of the dictionary
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
        }
    }.get(model_name, {"desc": "Standard ML Classifier", "strengths": ["Standard Performance"], "weaknesses": ["N/A"]})

# =========================
# PLOTLY GLASS RENDER
# =========================
def render_glass_importance_chart(df: pd.DataFrame):
    top_df = df.head(10).sort_values(by="importance", ascending=True)
    
    fig = px.bar(
        top_df, 
        x="importance", 
        y="feature",
        orientation='h',
        color="importance",
        color_continuous_scale="Viridis",
        labels={"importance": "Impact Score", "feature": "Diagnostic Feature"}
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        height=400,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =========================
# MAIN
# =========================
def main():
    render_page_header("Model Insights", "Deep-dive into algorithm behavior, strengths, and diagnostic logic.")

    try:
        best_bundle = load_model_bundle(BEST_MODEL_PATH)
        
        render_section_title("Current Champion Model")
        c1, c2, c3 = st.columns(3)
        with c1: render_metric_card("Best Model", best_bundle["model_name"])
        with c2: render_metric_card("Scaling", "Required" if best_bundle["scaling_required"] else "None")
        with c3: render_metric_card("Feature Set", f"{len(best_bundle['feature_names'])} Inputs")

        render_divider()
        st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
        render_section_title("Compare Algorithms")
        model_options = ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors"]
        selected_model = st.selectbox("Select a model to inspect:", model_options)
        
        if selected_model == best_bundle["model_name"]:
            st.info(f"✨ **{selected_model}** is currently the top-performing model.")
        st.markdown('</div>', unsafe_allow_html=True)

        selected_bundle = load_selected_model_bundle(selected_model)
        info = get_model_info(selected_model)

        st.write("")
        render_section_title(f"{selected_model} Profile")
        render_card("Technical Summary", info.get("desc", ""))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="apple-glass" style="border-left: 5px solid #34d399;">', unsafe_allow_html=True)
            st.markdown("**Core Strengths**")
            for s in info.get("strengths", []):
                st.markdown(f"✅ {s}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="apple-glass" style="border-left: 5px solid #f87171;">', unsafe_allow_html=True)
            st.markdown("**Known Limitations**")
            for w in info.get("weaknesses", []):
                st.markdown(f"⚠️ {w}")
            st.markdown('</div>', unsafe_allow_html=True)

        render_divider()
        render_section_title("Diagnostic Feature Importance")
        
        importance_df = get_feature_importance(selected_bundle)

        if importance_df is None or importance_df.empty:
            st.warning("Feature importance is mathematically unavailable for this model type.")
        else:
            st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
            render_glass_importance_chart(importance_df)
            st.markdown(f"**Insight:** Top driver: **{importance_df.iloc[0]['feature']}**.")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Analysis Error: {e}")

    render_divider()
    col_a, col_b = st.columns(2)
    with col_a:
        render_card("🔍 Why Explainability?", "AI transparency allows verification of correct biological markers.")
    with col_b:
        render_card("🏆 Selection Criteria", "Balanced evaluation based on Accuracy and F1-Score.")

    render_footer_note("© 2026 Breast Cancer Detection AI")

if __name__ == "__main__":
    main()