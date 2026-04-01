"""
5_About_Project.py

Enhanced UI version of About Page
"""

import streamlit as st


def main():
    # =========================
    # HERO SECTION
    # =========================
    st.set_page_config(page_title="About Project", layout="wide")

    st.title("🧬 Breast Cancer Detection AI")
    st.markdown(
        "### AI-powered decision-support system for early-stage breast cancer classification"
    )
    st.divider()

    # =========================
    # OVERVIEW CARDS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Models Used", "5")
    col2.metric("Features", "30")
    col3.metric("Prediction Type", "Binary Classification")

    st.divider()

    # =========================
    # PROJECT STORY
    # =========================
    st.subheader("📌 What is this project?")

    st.info(
        "This system uses supervised machine learning to classify breast tumors "
        "as **benign or malignant** based on diagnostic features extracted from medical data."
    )

    st.write(
        "Built as part of a Master's thesis, this project demonstrates how AI can be "
        "integrated into an interactive web application for medical decision support."
    )

    # =========================
    # OBJECTIVE (HIGHLIGHT)
    # =========================
    st.subheader("🎯 Objective")

    st.success(
        "To build an interpretable and scalable AI system that assists in early breast cancer detection."
    )

    # =========================
    # PIPELINE VISUAL FLOW
    # =========================
    st.subheader("⚙️ How the System Works")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown("📥 **Data Input**")
    col2.markdown("🧹 **Preprocessing**")
    col3.markdown("🤖 **Model Training**")
    col4.markdown("📊 **Evaluation**")
    col5.markdown("🚀 **Deployment**")

    st.divider()

    # =========================
    # MODELS SECTION
    # =========================
    st.subheader("🤖 Models Used")

    cols = st.columns(5)

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN"
    ]

    for col, model in zip(cols, models):
        col.success(model)

    # =========================
    # DATASET SECTION
    # =========================
    st.divider()
    st.subheader("📊 Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Breast Cancer Wisconsin Dataset**")
        st.write("✔ 30 numerical features")
        st.write("✔ Structured tabular data")

    with col2:
        st.write("**Target Classes**")
        st.success("Benign")
        st.error("Malignant")

    # =========================
    # FEATURES GRID
    # =========================
    st.divider()
    st.subheader("🚀 Application Features")

    col1, col2 = st.columns(2)

    with col1:
        st.write("✔ Manual Prediction")
        st.write("✔ Batch CSV Prediction")
        st.write("✔ Model Comparison")
        st.write("✔ PCA Visualization")

    with col2:
        st.write("✔ Feature Importance")
        st.write("✔ ROC Curve")
        st.write("✔ Confusion Matrix")
        st.write("✔ Model Insights Dashboard")

    # =========================
    # LIMITATIONS
    # =========================
    st.divider()
    st.subheader("⚠️ Limitations")

    st.warning(
        """
        - Does not use real patient clinical data  
        - Limited to structured numerical dataset  
        - Some models lack explainability (SVM, KNN)  
        """
    )

    # =========================
    # FUTURE SCOPE
    # =========================
    st.divider()
    st.subheader("🚀 Future Scope")

    col1, col2 = st.columns(2)

    with col1:
        st.write("🔹 AI-generated medical reports")
        st.write("🔹 Explainable AI (XAI)")
        st.write("🔹 Clinical dashboard")

    with col2:
        st.write("🔹 Patient data integration")
        st.write("🔹 Ensemble learning")
        st.write("🔹 Real-world deployment")

    # =========================
    # DISCLAIMER
    # =========================
    st.divider()
    st.subheader("❗ Disclaimer")

    st.error(
        "This application is for research purposes only and is NOT a medical diagnostic tool."
    )

   


if __name__ == "__main__":
    main()