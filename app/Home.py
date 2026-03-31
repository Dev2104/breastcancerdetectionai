"""
1_Home.py

Home page for the Breast Cancer Detection AI Streamlit application.

This page provides:
- project overview
- current model information from the saved deployment bundle
- app usage guidance
- current capabilities
- research disclaimer
"""

from pathlib import Path
import sys

import streamlit as st

# Ensure project root is available for imports when running inside
# the Streamlit multipage application structure.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle  # noqa: E402


def render_model_information() -> None:
    """
    Load the saved model bundle and display core model metadata.

    Returns
    -------
    None
    """
    st.subheader("Current Model Information")

    try:
        model_bundle = load_model_bundle()
        model_name = model_bundle["model_name"]
        scaling_required = model_bundle["scaling_required"]
        feature_names = model_bundle["feature_names"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Name", model_name)
        with col2:
            st.metric("Scaling Required", "Yes" if scaling_required else "No")
        with col3:
            st.metric("Number of Features", len(feature_names))

    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first by running: python -m src.train")
    except (KeyError, ValueError) as exc:
        st.error(f"Model bundle could not be read correctly: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error while loading model information: {exc}")


def render_project_overview() -> None:
    """
    Display the project overview section.

    Returns
    -------
    None
    """
    st.title("Breast Cancer Detection AI")

    st.write(
        "This application predicts whether a breast tumor is **benign** or "
        "**malignant** using **supervised machine learning** models trained on "
        "diagnostic breast cancer data."
    )

    st.write(
        "It is designed as a **research and educational decision-support "
        "prototype** to demonstrate how artificial intelligence can assist in "
        "structured clinical data analysis for early breast cancer detection."
    )


def render_how_to_use_section() -> None:
    """
    Display guidance on how to use the multipage application.

    Returns
    -------
    None
    """
    st.subheader("How to Use This App")

    st.markdown(
        """
        **Home**  
        View the project overview, current model details, app capabilities, and usage guidance.

        **Manual Prediction**  
        Enter all required diagnostic feature values manually to generate a single tumor classification result.

        **Batch Prediction**  
        Upload a CSV file containing multiple patient records and generate predictions for all rows at once.

        **Visualizations**  
        Explore dataset and model-related visual outputs such as distributions, comparison plots, and evaluation charts.

        **Model Insights**  
        Review model-related insights such as the selected model, feature importance, and interpretability-oriented outputs.

        **About Project**  
        Read the methodology, thesis alignment, project purpose, and research context behind the application.
        """
    )


def render_capabilities_section() -> None:
    """
    Display the current capabilities of the application.

    Returns
    -------
    None
    """
    st.subheader("Current Capabilities")

    st.markdown(
        """
        - Manual single prediction
        - CSV batch prediction
        - Probability output for supported models
        - Deployment-ready saved model bundle
        """
    )


def render_disclaimer() -> None:
    """
    Display the research and medical-use disclaimer.

    Returns
    -------
    None
    """
    st.subheader("Disclaimer")

    st.warning(
        "This application is for research and educational purposes only. "
        "It is not a medical diagnosis system. Users should consult qualified "
        "healthcare professionals for real medical decisions."
    )


def main() -> None:
    """
    Main entry point for the Home page.

    Returns
    -------
    None
    """
    render_project_overview()
    st.divider()
    render_model_information()
    st.divider()
    render_how_to_use_section()
    st.divider()
    render_capabilities_section()
    st.divider()
    render_disclaimer()


if __name__ == "__main__":
    main()