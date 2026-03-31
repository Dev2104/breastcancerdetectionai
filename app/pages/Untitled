"""
streamlit_app.py

Streamlit web application for breast cancer prediction.

Features:
1. Manual single prediction using all required model features
2. CSV upload for batch prediction
3. Model metadata display
4. Downloadable batch prediction results

Run with:
    streamlit run app/streamlit_app.py
"""

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is available for imports when running:
# streamlit run app/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (  # noqa: E402
    load_model_bundle,
    predict_batch,
    predict_single,
)


st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🩺",
    layout="wide",
)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for download.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert.

    Returns
    -------
    bytes
        CSV content as bytes.
    """
    return df.to_csv(index=False).encode("utf-8")


def build_manual_input_form(feature_names: list[str]) -> dict:
    """
    Build Streamlit input fields for all required features.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list of feature names required by the model.

    Returns
    -------
    dict
        Dictionary containing user input values for each feature.
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

    Parameters
    ----------
    result : dict
        Output dictionary returned by predict_single().

    Returns
    -------
    None
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


def render_manual_prediction(feature_names: list[str]) -> None:
    """
    Render the manual single-prediction interface.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list of feature names required by the model.

    Returns
    -------
    None
    """
    st.subheader("Manual Single Prediction")
    st.write(
        "Enter values for all diagnostic features below, then click "
        "**Predict** to classify the tumor as benign or malignant."
    )

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


def render_csv_prediction(feature_names: list[str]) -> None:
    """
    Render the batch prediction interface for CSV uploads.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list of feature names required by the model.

    Returns
    -------
    None
    """
    st.subheader("Batch Prediction via CSV Upload")
    st.write(
        "Upload a CSV file containing all required feature columns. "
        "The app will generate predictions for every row."
    )

    with st.expander("View required CSV columns"):
        st.write(feature_names)

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="The CSV must include all required feature columns."
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                st.error("The uploaded CSV file is empty.")
                return

            missing_columns = [col for col in feature_names if col not in input_df.columns]
            if missing_columns:
                st.error(
                    "The uploaded CSV is missing required columns: "
                    f"{missing_columns}"
                )
                return

            results_df = predict_batch(input_df)

            st.success("Batch prediction completed successfully.")
            st.subheader("Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = dataframe_to_csv_bytes(results_df)
            st.download_button(
                label="Download Results as CSV",
                data=csv_bytes,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
            )

        except pd.errors.EmptyDataError:
            st.error("The uploaded file could not be read because it is empty.")
        except ValueError as exc:
            st.error(f"Input validation error: {exc}")
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"An unexpected error occurred while processing the CSV: {exc}")


def main() -> None:
    """
    Main Streamlit application entry point.

    Returns
    -------
    None
    """
    st.title("Breast Cancer Detection AI")
    st.write(
        "This application predicts whether a breast tumor is **benign** or "
        "**malignant** using a trained supervised machine learning model."
    )

    try:
        model_bundle = load_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first by running: python -m src.train")
        return
    except ValueError as exc:
        st.error(f"Model bundle validation failed: {exc}")
        return
    except Exception as exc:
        st.error(f"Unable to load model bundle: {exc}")
        return

    model_name = model_bundle["model_name"]
    scaling_required = model_bundle["scaling_required"]
    feature_names = model_bundle["feature_names"]

    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Model Name:** {model_name}")
        st.write(f"**Scaling Required:** {'Yes' if scaling_required else 'No'}")
        st.write(f"**Number of Features:** {len(feature_names)}")

    prediction_mode = st.radio(
        "Choose Prediction Mode",
        options=["Manual Single Prediction", "CSV Batch Prediction"],
        horizontal=True,
    )

    if prediction_mode == "Manual Single Prediction":
        render_manual_prediction(feature_names)
    else:
        render_csv_prediction(feature_names)


if __name__ == "__main__":
    main()