from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle  # noqa: E402


st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🩺",
    layout="wide",
)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .hero-box {
            padding: 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 700;
            color: white;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.92;
            color: white;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .feature-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
            min-height: 120px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }

        .feature-card h4 {
            margin: 0 0 0.45rem 0;
            color: #111827 !important;
            font-size: 1.02rem;
            font-weight: 700;
        }

        .feature-card p {
            margin: 0;
            color: #374151 !important;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .project-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1rem;
            min-height: 110px;
        }

        .project-card p {
            margin: 0;
            color: #1f2937 !important;
            font-size: 0.98rem;
            line-height: 1.6;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">🧬 Breast Cancer Detection AI</div>
            <div class="hero-subtitle">
                AI-powered decision-support system for early breast cancer classification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_overview():
    st.markdown('<div class="section-title">Current Model</div>', unsafe_allow_html=True)

    try:
        model_bundle = load_model_bundle()

        col1, col2, col3 = st.columns(3)
        col1.metric("Model", model_bundle["model_name"])
        col2.metric("Scaling", "Yes" if model_bundle["scaling_required"] else "No")
        col3.metric("Features", len(model_bundle["feature_names"]))

    except Exception:
        st.warning("Model not loaded. Please run training first.")


def render_project_info():
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="project-card">
                <p>
                    This system predicts whether a tumor is <b>benign</b> or <b>malignant</b>
                    using supervised machine learning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="project-card">
                <p>
                    Designed as a research-level decision-support system for AI in
                    healthcare applications.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_features():
    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <h4>✍️ Manual Prediction</h4>
                <p>Enter patient feature values manually and receive an instant AI-based prediction.</p>
            </div>

            <div class="feature-card">
                <h4>📂 Batch Prediction</h4>
                <p>Upload a CSV file and generate predictions for multiple patient cases at once.</p>
            </div>

            <div class="feature-card">
                <h4>📊 Visualizations</h4>
                <p>Explore class distribution, PCA, confusion matrix, ROC curve, and other analytical views.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <h4>🔍 Model Insights</h4>
                <p>Understand model behavior, compare algorithms, and review important diagnostic features.</p>
            </div>

            <div class="feature-card">
                <h4>⚙️ ML Pipeline</h4>
                <p>Automated training, evaluation, model comparison, and deployment-ready model packaging.</p>
            </div>

            <div class="feature-card">
                <h4>🚀 Future Ready</h4>
                <p>Structured to support richer clinical insights, report generation, and decision-support upgrades.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_workflow():
    st.markdown('<div class="section-title">How to Use This App</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.info("**1. Home**\n\nOverview of the system")
        st.info("**2. Manual Prediction**\n\nPredict single case")
        st.info("**3. Batch Prediction**\n\nUpload CSV")

    with col2:
        st.info("**4. Visualizations**\n\nExplore data")
        st.info("**5. Model Insights**\n\nUnderstand model")
        st.info("**6. About Project**\n\nFull details")


def render_disclaimer():
    st.markdown('<div class="section-title">Disclaimer</div>', unsafe_allow_html=True)

    st.warning(
        "This application is for research purposes only and is NOT a medical diagnostic tool."
    )


def render_footer():
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            **Dev Tailor**  
            Aspiring Data Scientist | AI & Analytics  

            🧠 Focus: AI in Healthcare
            """
        )

    with col2:
        st.markdown(
            """
            <a href="https://www.linkedin.com/in/dev-tailor1212" target="_blank">🔗 LinkedIn</a><br>
            <a href="https://github.com/Dev2104/breastcancerdetectionai" target="_blank">💻 GitHub</a><br>
            📧 tailordev663@gmail.com
            """,
            unsafe_allow_html=True,
        )

    st.caption("© 2026 Breast Cancer Detection AI")


def main():
    inject_custom_css()
    render_hero()
    render_model_overview()

    st.divider()
    render_project_info()

    st.divider()
    render_features()

    st.divider()
    render_workflow()

    st.divider()
    render_disclaimer()

    render_footer()


if __name__ == "__main__":
    main()