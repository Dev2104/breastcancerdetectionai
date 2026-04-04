from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model_bundle, predict_batch  # noqa: E402
from UI.ui_master import (
    configure_page,
    inject_master_theme,
    render_page_header,
    render_section_title,
    render_metric_card,
    render_divider,
    render_footer_note,
    render_info_banner
)

# 1. Setup Page & Glass Theme
configure_page("Batch Prediction | Breast Cancer AI")
inject_master_theme()

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def render_summary_cards(results_df: pd.DataFrame) -> None:
    """
    Display summary statistics in Apple Glass tiles.
    """
    total_rows = len(results_df)
    benign_count = int((results_df["prediction_label"] == "benign").sum())
    malignant_count = int((results_df["prediction_label"] == "malignant").sum())

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Processed", str(total_rows))
    with col2:
        render_metric_card("Benign Cases", str(benign_count))
    with col3:
        render_metric_card("Malignant Cases", str(malignant_count))

def main() -> None:
    # 1. Header Section
    render_page_header(
        "Batch Prediction", 
        "Upload clinical datasets in CSV format for high-volume AI diagnostic classification."
    )

    try:
        model_bundle = load_model_bundle()
        feature_names = model_bundle["feature_names"]
    except Exception as exc:
        st.error(f"Model Load Error: {exc}")
        return

    # 2. Sidebar Glass Info
    with st.sidebar:
        st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
        st.subheader("📊 Batch Config")
        st.write(f"**Model:** {model_bundle['model_name']}")
        st.write(f"**Required Columns:** {len(feature_names)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Upload Section (Wrapped in Glass)
    st.markdown('<div class="apple-glass">', unsafe_allow_html=True)
    
    with st.expander("📂 View Required CSV Schema"):
        st.write("Your CSV must include the following columns:")
        st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader(
        "Drop your clinical data here",
        type=["csv"],
        help="Ensure all 30 diagnostic features are present as columns."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        render_info_banner("Awaiting CSV upload to begin automated analysis.")
        render_divider()
        render_footer_note("© 2026 Breast Cancer Detection AI")
        return

    # 4. Processing & Results
    try:
        input_df = pd.read_csv(uploaded_file)

        if input_df.empty:
            st.error("The uploaded CSV file contains no data.")
            return

        # Preview Section
        render_section_title("Data Preview")
        st.dataframe(input_df.head(3), use_container_width=True)

        # Validation
        missing_columns = [col for col in feature_names if col not in input_df.columns]
        if missing_columns:
            st.error(f"Missing Required Columns: {missing_columns}")
            return

        # Prediction Logic
        with st.spinner("Processing Batch Analytics..."):
            results_df = predict_batch(input_df)

        st.success("Batch Analysis Complete!")
        
        # Summary Glass Metrics
        render_section_title("Diagnostic Summary")
        render_summary_cards(results_df)

        # Full Results Table
        render_section_title("Detailed Results")
        st.dataframe(results_df, use_container_width=True)

        # Download Section
        render_divider()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            csv_bytes = dataframe_to_csv_bytes(results_df)
            st.download_button(
                label="🚀 Download Prediction Report (CSV)",
                data=csv_bytes,
                file_name="ai_diagnostic_report.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as exc:
        st.error(f"An error occurred during processing: {exc}")

    render_divider()
    render_footer_note("© 2026 Breast Cancer Detection AI | Batch Processing Module")

if __name__ == "__main__":
    main()