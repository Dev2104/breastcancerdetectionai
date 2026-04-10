import streamlit as st
from UI.ui_master import (
    configure_page,
    inject_master_theme,
    render_page_header,
    render_section_title,
    render_metric_card,
    render_divider,
    render_footer_note
)

def main():
    # 1. Setup Page & Theme
    configure_page("About | Breast Cancer Detection AI")
    inject_master_theme()

    # 2. Hero Section
    render_page_header(
        "🧬 Project Intelligence", 
        "A deep-dive into the architecture, methodology, and scope of the AI diagnostic system."
    )

    # 3. Quick Stats
    c1, c2, c3 = st.columns(3)
    with c1: render_metric_card("Algorithms", "5 Models")
    with c2: render_metric_card("Data Points", "30 Features")
    with c3: render_metric_card("Objective", "Binary Classification")

    render_divider()

    # 4. Project Narrative
    st.markdown(
        """
        <div class="apple-glass">
            <h3 style="margin-top:0; color: #1fa8bb;">📌 Project Narrative</h3>
            <p style="opacity:0.8; line-height: 1.6;">
                This system leverages supervised machine learning to classify breast tumors as <b>Benign</b> or <b>Malignant</b>. 
                Developed as a core part of a Master's thesis, it bridges the gap between complex ML models and clinical decision-support interfaces.
            </p>
            <div style="background: rgba(31, 168, 187, 0.15); padding: 12px; border-radius: 12px; border-left: 4px solid #1fa8bb;">
                🎯 <b>Goal:</b> To provide an interpretable, scalable AI system for early-stage detection support.
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # 5. Pipeline Visualization (Sanitized HTML)
    render_divider()
    render_section_title("⚙️ System Pipeline")
    
    # We build the HTML block manually to ensure no f-string leaks
    pipeline_html = """
    <div style="display: flex; flex-wrap: wrap; gap: 12px; justify-content: space-between; margin-bottom: 20px;">
        <div class="apple-glass" style="flex: 1; min-width: 160px; text-align: center;">
            <div style="font-weight: 700; color: var(--text-color);">📥 Data Input</div>
            <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.6;">WISCONSIN Dataset</div>
        </div>
        <div class="apple-glass" style="flex: 1; min-width: 160px; text-align: center;">
            <div style="font-weight: 700; color: var(--text-color);">🧹 Preprocessing</div>
            <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.6;">Scaling & Cleaning</div>
        </div>
        <div class="apple-glass" style="flex: 1; min-width: 160px; text-align: center;">
            <div style="font-weight: 700; color: var(--text-color);">🤖 Training</div>
            <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.6;">Multi-model Fit</div>
        </div>
        <div class="apple-glass" style="flex: 1; min-width: 160px; text-align: center;">
            <div style="font-weight: 700; color: var(--text-color);">📊 Evaluation</div>
            <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.6;">Metrics & Curves</div>
        </div>
        <div class="apple-glass" style="flex: 1; min-width: 160px; text-align: center;">
            <div style="font-weight: 700; color: var(--text-color);">🚀 Deployment</div>
            <div style="font-size: 0.85rem; color: var(--text-color); opacity: 0.6;">Streamlit UI</div>
        </div>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # 6. Technical Stack
    render_divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            <div class="apple-glass" style="min-height: 250px;">
                <h3 style="margin-top:0; color: #1fa8bb;">🤖 Model Library</h3>
                <ul style="list-style-type: none; padding-left: 0; opacity: 0.9; line-height: 2;">
                    <li>🔹 Logistic Regression</li>
                    <li>🔹 Decision Tree</li>
                    <li>🔹 Random Forest</li>
                    <li>🔹 SVM (Kernel Support)</li>
                    <li>🔹 K-Nearest Neighbors</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

    with col_right:
        st.markdown(
            """
            <div class="apple-glass" style="min-height: 250px;">
                <h3 style="margin-top:0; color: #1fa8bb;">📊 Dataset Schema</h3>
                <p style="margin-bottom: 10px;"><b>Wisconsin Diagnostic (WDBC)</b></p>
                <p style="font-size: 0.9rem; opacity: 0.8; margin: 0;">✔ 30 Numerical Features</p>
                <p style="font-size: 0.9rem; opacity: 0.8; margin: 0;">✔ Structured Tabular Format</p>
                <hr style="opacity: 0.15; margin: 15px 0;">
                <span style="color:#34d399; font-weight: 600;">● Benign Class</span> | 
                <span style="color:#f87171; font-weight: 600;">● Malignant Class</span>
            </div>
            """, unsafe_allow_html=True
        )

    # 7. Scope & Capabilities
    render_divider()
    render_section_title("🚀 Scope & Capabilities")
    
    cap_html = """
    <div style="display: flex; flex-wrap: wrap; gap: 15px;">
        <div class="apple-glass" style="flex: 1; min-width: 250px;">
            <h4 style="margin-top:0; color: #60a5fa;">Core Features</h4>
            <div style="font-size: 0.9rem; opacity: 0.8; line-height: 1.8;">
                ✔ Manual Entry Diagnostics<br>✔ Batch CSV Processing<br>✔ Interactive Visual Reports<br>✔ PCA Feature Analytics
            </div>
        </div>
        <div class="apple-glass" style="flex: 1; min-width: 250px;">
            <h4 style="margin-top:0; color: #60a5fa;">Future Scope</h4>
            <div style="font-size: 0.9rem; opacity: 0.8; line-height: 1.8;">
                🔹 Explainable AI (XAI) Integration<br>🔹 PDF Medical Report Export<br>🔹 Multi-model Stacking<br>🔹 Real-time Clinical API
            </div>
        </div>
    </div>
    """
    st.markdown(cap_html, unsafe_allow_html=True)

    # 8. Disclaimer
    st.write("")
    st.error("❗ **Legal Disclaimer:** This application is strictly for research and academic purposes. It is NOT a certified medical diagnostic tool.", icon="⚖️")

    render_footer_note("© 2026 Breast Cancer Detection AI")

if __name__ == "__main__":
    main()