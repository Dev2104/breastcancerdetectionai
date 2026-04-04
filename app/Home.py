from pathlib import Path
import sys
import streamlit as st

# Setup Pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle  # noqa: E402
from UI.ui_master import (  # noqa: E402
    configure_page,
    inject_master_theme,
    render_page_header,
    render_info_banner,
    render_section_title,
    render_card,
    render_metric_card,
    render_footer_note,
    render_divider,
)

# 1. Setup Page & Global Theme
configure_page("Breast Cancer Detection AI")
inject_master_theme()

def render_model_overview():
    render_section_title("Current Model Status")
    try:
        model_bundle = load_model_bundle()
        model_name = str(model_bundle.get("model_name", "Unknown"))
        scaling_required = "Yes" if model_bundle.get("scaling_required", False) else "No"
        feature_count = str(len(model_bundle.get("feature_names", [])))
    except Exception:
        model_name, scaling_required, feature_count = "N/A", "N/A", "N/A"
        st.warning("Model not loaded. Please run training first.")

    c1, c2, c3 = st.columns(3)
    with c1: render_metric_card("Selected Model", model_name)
    with c2: render_metric_card("Feature Scaling", scaling_required)
    with c3: render_metric_card("Input Features", feature_count)

def render_project_info():
    render_section_title("Project Overview")
    col1, col2 = st.columns(2)
    with col1:
        render_card("Clinical Prediction Focus", "Predicts tumor classification (Benign/Malignant) using supervised machine learning.")
    with col2:
        render_card("Research Application", "A decision-support tool combining AI, visual analytics, and model transparency.")

def render_features():
    render_section_title("Core Capabilities")
    col1, col2 = st.columns(2)
    with col1:
        render_card("✍️ Manual Prediction", "Input features manually for a real-time diagnostic classification.")
        render_card("📂 Batch Prediction", "Upload CSV data for high-volume automated screening.")
    with col2:
        render_card("📊 Visual Analytics", "Advanced charts including PCA, ROC Curves, and Confusion Matrices.")
        render_card("🔍 Model Insights", "Transparency into feature importance and decision-making logic.")

def render_workflow():
    render_section_title("Platform Workflow")
    
    steps = [
        ("1. Home", "Overview of model & system."),
        ("2. Manual", "Single-case diagnostics."),
        ("3. Batch", "CSV processing."),
        ("4. Visuals", "Analytical deep-dives."),
        ("5. Insights", "Model interpretation."),
        ("6. About", "Full technical scope.")
    ]
    
    cols = st.columns(3)
    # FIXED: Single loop to prevent duplication
    for i, (title, desc) in enumerate(steps):
        with cols[i % 3]:
            render_card(title, desc)

def render_footer():
    render_divider()
    render_section_title("Developer Contact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Styled to match your reference image exactly
        st.markdown(
            f"""
            <div class="apple-glass" style="min-height: 160px;">
                <div style="font-size: 1.3rem; font-weight: 700; color: var(--text-color); margin-bottom: 8px; display: flex; align-items: center; gap: 10px;">
                    💻 Dev Tailor
                </div>
                <p style="color: var(--text-color); opacity: 0.8; font-size: 0.95rem; margin-bottom: 12px;">
                    Aspiring Data Scientist | AI in Healthcare
                </p>
                <a href="mailto:tailordev663@gmail.com" style="display: flex; align-items: center; gap: 8px; font-weight: 600; color: #1fa8bb; text-decoration: none;">
                    📧 Email: tailordev663@gmail.com
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Styled to match your reference image exactly
        st.markdown(
            f"""
            <div class="apple-glass" style="min-height: 160px;">
                <div style="font-size: 1.3rem; font-weight: 700; color: var(--text-color); margin-bottom: 12px; display: flex; align-items: center; gap: 10px;">
                    🔗 Quick Links
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <a href="https://www.linkedin.com/in/dev-tailor1212" target="_blank" style="font-weight: 600; color: #1fa8bb; text-decoration: none;">
                        LinkedIn
                    </a>
                    <a href="https://github.com/Dev2104/breastcancerdetectionai" target="_blank" style="font-weight: 600; color: #1fa8bb; text-decoration: none;">
                        GitHub
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    render_footer_note("© 2026 Breast Cancer Detection AI")

def main():
    # Header Section
    render_page_header(
        "🧬 Breast Cancer Detection AI", 
        "Advanced diagnostic support system for early cancer classification."
    )
    
    render_info_banner("Welcome! This dashboard provides the core entry point for all diagnostic and analytical modules.")
    
    # Render all UI sections
    render_model_overview()
    render_project_info()
    render_features()
    render_workflow()
    
    # Disclaimer and Footer
    st.error("**Disclaimer:** This tool is for research purposes only and is not a substitute for professional medical advice.", icon="⚠️")
    render_footer()

if __name__ == "__main__":
    main()