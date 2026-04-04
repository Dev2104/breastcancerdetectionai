from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is available
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.explain import get_feature_importance
from src.unsupervised import run_pca_analysis

from UI.ui_master import (
    configure_page,
    inject_master_theme,
    render_page_header,
    render_divider,
    render_footer_note
)

# Config for static images (Confusion Matrix/ROC)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = FIGURES_DIR / "roc_curve.png"

configure_page("Visualizations | Breast Cancer AI")
inject_master_theme()

# --- THE SECRET SAUCE: GLASS THEME FOR PLOTLY ---
def apply_glass_layout(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        margin=dict(t=40, b=40, l=20, r=20),
        hoverlabel=dict(bgcolor="#13486f", font_size=14, font_family="SF Pro Display"),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
    )
    return fig

def render_glass_block(title, description, content_func, *args, **kwargs):
    st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    st.caption(description)
    content_func(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)

# --- NEW PLOTLY FUNCTIONS ---

def render_class_distribution(df):
    counts = df["target_label"].value_counts().reset_index()
    fig = px.bar(
        counts, x="target_label", y="count",
        color="target_label",
        color_discrete_map={"benign": "#1fa8bb", "malignant": "#60a5fa"},
        labels={"target_label": "Tumor Class", "count": "Patient Count"}
    )
    st.plotly_chart(apply_glass_layout(fig), use_container_width=True)

def render_feature_histograms(df):
    feature_columns = [col for col in df.columns if col not in {"target", "target_label"}]
    selected_feature = st.selectbox("Select Diagnostic Feature", options=feature_columns)
    
    fig = px.histogram(
        df, x=selected_feature, color="target_label",
        marginal="box", # Adds a box plot on top!
        color_discrete_map={"benign": "#1fa8bb", "malignant": "#60a5fa"},
        opacity=0.7, barmode="overlay"
    )
    st.plotly_chart(apply_glass_layout(fig), use_container_width=True)

def render_correlation_heatmap(df):
    numeric_df = df.drop(columns=["target", "target_label"]).copy()
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr, 
        text_auto=".2f",
        # CHANGE "mako" to "Viridis" (or "Ice" for a clinical look)
        color_continuous_scale="Viridis", 
        aspect="auto"
    )
    st.plotly_chart(apply_glass_layout(fig), use_container_width=True)

def render_pca_visualization():
    pca_df = run_pca_analysis()
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="target_label",
        color_discrete_map={"benign": "#1fa8bb", "malignant": "#60a5fa"},
        hover_data={"PC1": ":.2f", "PC2": ":.2f"}
    )
    # Highlight the clusters
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    st.plotly_chart(apply_glass_layout(fig), use_container_width=True)

def render_feature_importance():
    try:
        importance_df = get_feature_importance().head(12)
        fig = px.bar(
            importance_df, x="Absolute Importance", y="Feature",
            orientation='h', color="Absolute Importance",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(apply_glass_layout(fig), use_container_width=True)
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

def render_static_image(image_path: Path, msg: str):
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.info(msg)

# --- MAIN ---

def main():
    render_page_header("Visualizations", "Interactive AI diagnostics and clinical data analysis.")
    df = load_data()

    # Selector
    st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
    view_mode = st.selectbox("Select Analysis Module", [
        "Class Distribution", "Feature Histograms", "Correlation Heatmap", 
        "PCA Visualization", "Feature Importance", "Confusion Matrix", "ROC Curve"
    ])
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    if view_mode == "Class Distribution":
        render_glass_block("Dataset Balance", "Hover over bars to see exact counts.", render_class_distribution, df)
    elif view_mode == "Feature Histograms":
        render_glass_block("Feature Variance", "Use the boxplot on top to identify outliers.", render_feature_histograms, df)
    elif view_mode == "Correlation Heatmap":
        render_glass_block("Feature Relationships", "Interactive heatmap of diagnostic parameters.", render_correlation_heatmap, df)
    elif view_mode == "PCA Visualization":
        render_glass_block("Clustering Analysis", "2D projection showing inherent class separability.", render_pca_visualization)
    elif view_mode == "Feature Importance":
        render_glass_block("Model Drivers", "Relative importance of features in classification.", render_feature_importance)
    elif view_mode == "Confusion Matrix":
        render_glass_block("Confusion Matrix", "Static report generated during training.", render_static_image, CONFUSION_MATRIX_PATH, "Run training to generate.")
    elif view_mode == "ROC Curve":
        render_glass_block("ROC Curve", "Static performance report.", render_static_image, ROC_CURVE_PATH, "Run training to generate.")

    render_divider()
    render_footer_note("© 2026 Breast Cancer Detection AI | Powered by Plotly Interactive")

if __name__ == "__main__":
    main()