"""
3_Batch_Prediction.py

Batch prediction page for the Breast Cancer Detection AI Streamlit application.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle, predict_batch  # noqa: E402


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for download.
    """
    return df.to_csv(index=False).encode("utf-8")


def render_summary_cards(results_df: pd.DataFrame) -> None:
    """
    Display summary statistics for batch prediction results.
    """
    total_rows = len(results_df)
    benign_count = int((results_df["prediction_label"] == "benign").sum())
    malignant_count = int((results_df["prediction_label"] == "malignant").sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows Processed", total_rows)
    with col2:
        st.metric("Benign Count", benign_count)
    with col3:
        st.metric("Malignant Count", malignant_count)


def main() -> None:
    """
    Main entry point for the Batch Prediction page.
    """
    st.title("Batch Prediction")
    st.write(
        "Upload a CSV file containing all required diagnostic feature columns. "
        "The application will generate predictions for every row."
    )

    try:
        model_bundle = load_model_bundle()
        feature_names = model_bundle["feature_names"]
        model_name = model_bundle["model_name"]
        scaling_required = model_bundle["scaling_required"]
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Train the model first by running: python -m src.train")
        return
    except (KeyError, ValueError) as exc:
        st.error(f"Model bundle could not be read correctly: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error while loading the model bundle: {exc}")
        return

    with st.sidebar:
        st.header("Model Information")
        st.write(f"**Model Name:** {model_name}")
        st.write(f"**Scaling Required:** {'Yes' if scaling_required else 'No'}")
        st.write(f"**Number of Features:** {len(feature_names)}")

    with st.expander("View required CSV columns"):
        st.write(feature_names)

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="The CSV file must contain all required feature columns."
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin batch prediction.")
        return

    try:
        input_df = pd.read_csv(uploaded_file)

        if input_df.empty:
            st.error("The uploaded CSV file is empty.")
            return

        st.subheader("Uploaded Data Preview")
        st.dataframe(input_df.head(), use_container_width=True)

        missing_columns = [col for col in feature_names if col not in input_df.columns]
        if missing_columns:
            st.error(
                "The uploaded CSV is missing required columns: "
                f"{missing_columns}"
            )
            return

        results_df = predict_batch(input_df)

        st.success("Batch prediction completed successfully.")

        st.subheader("Summary")
        render_summary_cards(results_df)

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
        st.error("The uploaded file is empty and could not be read.")
    except ValueError as exc:
        st.error(f"Input validation error: {exc}")
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"An unexpected error occurred while processing the CSV: {exc}")


if __name__ == "__main__":
    main()