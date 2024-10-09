import streamlit as st

def apply_styles():
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] { gap: 3px; }
        .stTabs [data-baseweb="tab"] {
            height: 40px; white-space: pre-wrap; background-color: #13276F;
            border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px 2px; color: white;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF; color: #13276F; border: 2px solid #13276F; border-bottom: none;
        }
    </style>
    """, unsafe_allow_html=True)
