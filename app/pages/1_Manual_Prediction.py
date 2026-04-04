from pathlib import Path
import sys
import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle, predict_single  # noqa: E402
# IMPORT YOUR GLASS UI COMPONENTS
from UI.ui_master import (
    configure_page,
    inject_master_theme,
    render_page_header,
    render_section_title,
    render_card,
    render_metric_card,
    render_divider,
    render_footer_note
)

# 1. Setup Page & Glass Theme
configure_page("Manual Prediction | Breast Cancer AI")
inject_master_theme()

def build_manual_input_form(feature_names: list[str]) -> dict:
    """
    Build Streamlit input fields for all required features.
    """
    default_values = {
        "mean radius": 14.127, "mean texture": 19.290, "mean perimeter": 91.969,
        "mean area": 654.889, "mean smoothness": 0.096, "mean compactness": 0.104,
        "mean concavity": 0.089, "mean concave points": 0.049, "mean symmetry": 0.181,
        "mean fractal dimension": 0.063, "radius error": 0.405, "texture error": 1.217,
        "perimeter error": 2.866, "area error": 40.337, "smoothness error": 0.007,
        "compactness error": 0.025, "concavity error": 0.032, "concave points error": 0.012,
        "symmetry error": 0.021, "fractal dimension error": 0.004, "worst radius": 16.269,
        "worst texture": 25.677, "worst perimeter": 107.261, "worst area": 880.583,
        "worst smoothness": 0.132, "worst compactness": 0.254, "worst concavity": 0.272,
        "worst concave points": 0.115, "worst symmetry": 0.290, "worst fractal dimension": 0.084,
    }

    input_values = {}
    
    # We wrap the inputs in a transparent glass-like layout
    columns = st.columns(3)
    for index, feature in enumerate(feature_names):
        column = columns[index % 3]
        with column:
            input_values[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                value=float(default_values.get(feature, 0.0)),
                format="%.6f",
            )
    return input_values

def render_prediction_result(result: dict) -> None:
    """
    Display single prediction results in the Apple Glass format.
    """
    render_divider()
    render_section_title("Diagnostic Result")

    label = result["prediction_label"].upper()
    is_malignant = result["prediction_label"] == "malignant"
    
    # Use a high-visibility Glass Card for the main result
    bg_color = "rgba(255, 75, 75, 0.1)" if is_malignant else "rgba(40, 167, 69, 0.1)"
    border_color = "#ff4b4b" if is_malignant else "#28a745"
    
    st.markdown(
        f"""
        <div class="apple-glass" style="background: {bg_color}; border: 2px solid {border_color}; text-align: center;">
            <h2 style="margin: 0; color: {border_color};">PREDICTED CLASS: {label}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("") # Spacer

    # Metric Row
    m1, m2 = st.columns(2)
    with m1:
        render_metric_card("Model Confidence", f"{result['prediction']:.4f}")
    with m2:
        render_metric_card("Algorithm", result["model_name"])

    # Probability Glass Section
    probabilities = result.get("probabilities")
    if probabilities:
        st.write("")
        render_section_title("Detailed Probabilities")
        p1, p2 = st.columns(2)
        with p1:
            render_metric_card("Malignant Score", f"{probabilities['malignant']:.2%}")
        with p2:
            render_metric_card("Benign Score", f"{probabilities['benign']:.2%}")

def main() -> None:
    # 1. Header
    render_page_header(
        "Manual Prediction", 
        "Enter diagnostic feature values to generate a tumor classification using the trained AI model."
    )

    try:
        model_bundle = load_model_bundle()
        feature_names = model_bundle["feature_names"]
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return

    # 2. Sidebar Info (Glass Style)
    with st.sidebar:
        st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
        st.subheader("⚙️ Model Metadata")
        st.write(f"**Model:** {model_bundle['model_name']}")
        st.write(f"**Inputs:** {len(feature_names)} Features")
        st.write(f"**Scaling:** {'Active' if model_bundle['scaling_required'] else 'Inactive'}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Main Form
    # We use a custom div to wrap the form in a glass look
    st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
    with st.form("manual_prediction_form", border=False): # border=False because apple-glass provides it
        input_values = build_manual_input_form(feature_names)
        
        # Center the button
        _, btn_col, _ = st.columns([1, 1, 1])
        with btn_col:
            submitted = st.form_submit_button("Generate Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Result Rendering
    if submitted:
        with st.spinner("AI Analysis in progress..."):
            try:
                result = predict_single(input_values)
                render_prediction_result(result)
            except Exception as exc:
                st.error(f"Prediction Error: {exc}")
    
    render_divider()
    render_footer_note("© 2026 Breast Cancer Detection AI | Research Decision Support")

if __name__ == "__main__":
    main()