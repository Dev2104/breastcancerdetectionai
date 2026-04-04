from __future__ import annotations
import streamlit as st

def configure_page(title: str, icon: str = "🩺", layout: str = "wide") -> None:
    """Configures the Streamlit page settings."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded",
    )

def inject_master_theme() -> None:
    """Injects the Apple-inspired Glassmorphism CSS theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600&display=swap');

        /* 1. GLOBAL APP BACKGROUND */
        .stApp {
            background: radial-gradient(circle at top left, rgba(31, 168, 187, 0.12), transparent 40%),
                        radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.12), transparent 40%),
                        var(--background-color);
        }

        /* 2. THE GLASS CARD (The Fix for Gaps and Transparency) */
        .apple-glass {
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(15px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(15px) saturate(160%) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 20px !important;
            padding: 22px !important;
            margin-bottom: 15px !important;
            display: block !important;
            width: 100% !important;
            color: var(--text-color) !important;
        }

        /* Light Mode Visibility Overwrite */
        [data-theme="light"] .apple-glass {
            background: rgba(0, 0, 0, 0.04) !important;
            border: 1px solid rgba(0, 0, 0, 0.08) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
        }

        /* 3. TYPOGRAPHY & HEADERS */
        .glass-title {
            background: linear-gradient(90deg, #1fa8bb, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-family: 'SF Pro Display', sans-serif;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 0.5rem;
        }

        /* 4. SIDEBAR (Frosted Glass) */
        [data-testid="stSidebar"] {
            background: rgba(8, 44, 74, 0.8) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* 5. NATIVE BUTTONS (Apple Blue) */
        .stButton > button {
            background: #007AFF !important;
            color: white !important;
            border-radius: 12px !important;
            border: none !important;
            padding: 0.5rem 2rem !important;
            transition: all 0.3s ease;
            font-weight: 600 !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px rgba(0, 122, 255, 0.4);
            background: #0063d1 !important;
        }

        /* 6. INPUTS & SELECTBOXES */
        .stSelectbox, .stNumberInput {
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Renders a modern gradient header."""
    st.markdown(
        f"""
        <div class="apple-glass" style="border: none !important; background: transparent !important; backdrop-filter: none !important; padding: 0 !important;">
            <h1 class="glass-title">{title}</h1>
            <p style="color: var(--text-color); opacity: 0.7; font-size: 1.15rem; margin-top: -10px;">
                {subtitle if subtitle else ""}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_section_title(title: str) -> None:
    """Renders a standardized section title."""
    st.markdown(f"### {title}")

def render_info_banner(text: str) -> None:
    """Renders a standard Streamlit info banner."""
    st.info(text)

def render_card(title: str, body: str) -> None:
    """Renders a standard Apple Glass content card."""
    st.markdown(
        f"""
        <div class="apple-glass">
            <h4 style="color: #1fa8bb; margin-top: 0;">{title}</h4>
            <div style="color: var(--text-color); opacity: 0.85; line-height: 1.6;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_metric_card(label: str, value: str) -> None:
    """Renders a centered glass metric card."""
    st.markdown(
        f"""
        <div class="apple-glass" style="text-align: center;">
            <div style="color: var(--text-color); opacity: 0.6; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;">
                {label}
            </div>
            <div style="color: var(--text-color); font-size: 2.2rem; font-weight: 700;">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_divider() -> None:
    """Renders a subtle horizontal divider."""
    st.divider()

def render_footer_note(text: str) -> None:
    """Renders a small caption at the bottom of the page."""
    st.markdown(
        f"<p style='text-align: center; opacity: 0.5; font-size: 0.8rem; margin-top: 50px;'>{text}</p>", 
        unsafe_allow_html=True
    )