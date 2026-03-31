"""
2_Manual_Prediction.py

Manual prediction page for the Breast Cancer Detection AI Streamlit application.
"""

from pathlib import Path
import sys

import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle, predict_single  # noqa: E402


def build_manual_input_form(feature_names: list[str]) -> dict:
    """
    Build Streamlit input fields for all required features.
    """
    default_values = {
        "mean radius": 14.127,
        "mean texture": 19.290,
        "mean perimeter": 91.969,
        "mean area": 654.889,
        "mean smoothness": 0.096,
        "mean compactness": 0.104,
        "mean concavity": 0.089,
        "mean concave points": 0.049,
        "mean symmetry": 0.181,
        "mean fractal dimension": 0.063,
        "radius error": 0.405,
        "texture error": 1.217,
        "perimeter error": 2.866,
        "area error": 40.337,
        "smoothness error": 0.007,
        "compactness error": 0.025,
        "concavity error": 0.032,
        "concave points error": 0.012,
        "symmetry error": 0.021,
        "fractal dimension error": 0.004,
        "worst radius": 16.269,
        "worst texture": 25.677,
        "worst perimeter": 107.261,
        "worst area": 880.583,
        "worst smoothness": 0.132,
        "worst compactness": 0.254,
        "worst concavity": 0.272,
        "worst concave points": 0.115,
        "worst symmetry": 0.290,
        "worst fractal dimension": 0.084,
    }

    input_values = {}

    columns = st.columns(3)
    for index, feature in enumerate(feature_names):
        column = columns[index % 3]
        with column:
            input_values[feature] = st.number_input(
                label=feature,
                value=float(default_values.get(feature, 0.0)),
                format="%.6f",
            )

    return input_values


def render_prediction_result(result: dict) -> None:
    """
    Display single prediction results in a clean format.
    """
    st.subheader("Prediction Result")

    label = result["prediction_label"]
    numeric_prediction = result["prediction"]

    if label == "malignant":
        st.error(f"Predicted Class: {label.capitalize()}")
    else:
        st.success(f"Predicted Class: {label.capitalize()}")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Numeric Prediction", numeric_prediction)
    with metric_col2:
        st.metric("Model Used", result["model_name"])

    probabilities = result.get("probabilities")
    if probabilities is not None:
        st.subheader("Prediction Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Malignant Probability", f"{probabilities['malignant']:.4f}")
        with prob_col2:
            st.metric("Benign Probability", f"{probabilities['benign']:.4f}")


def main() -> None:
    """
    Main entry point for the Manual Prediction page.
    """
    st.title("Manual Prediction")
    st.write(
        "Enter values for all required diagnostic features below, then click "
        "**Predict** to classify the tumor as benign or malignant."
    )

    try:
        model_bundle = load_model_bundle()
        model_name = model_bundle["model_name"]
        scaling_required = model_bundle["scaling_required"]
        feature_names = model_bundle["feature_names"]
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first by running: python -m src.train")
        return
    except (KeyError, ValueError) as exc:
        st.error(f"Model bundle could not be read correctly: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error while loading model information: {exc}")
        return

    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Model Name:** {model_name}")
        st.write(f"**Scaling Required:** {'Yes' if scaling_required else 'No'}")
        st.write(f"**Number of Features:** {len(feature_names)}")

    with st.form("manual_prediction_form"):
        input_values = build_manual_input_form(feature_names)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            result = predict_single(input_values)
            render_prediction_result(result)
        except ValueError as exc:
            st.error(f"Input validation error: {exc}")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"An unexpected error occurred during prediction: {exc}")


if __name__ == "__main__":
    main()