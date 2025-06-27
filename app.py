# app.py
import streamlit as st
import pandas as pd
from dashboard_page import show_dashboard
from compare_page import show_compare
from map import show_map

st.set_page_config(page_title="CLIMATE DASHBOARD", layout="wide")
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 55px;
        font-weight: bold;
        margin-top: 5px;
        margin-bottom: 5px;
        color: #1a1a1a;
        font-family: 'Georgia', serif;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #333;
        font-style: italic;
        margin-bottom: 30px;
        font-family: 'Georgia', serif;
    }
    </style>

    <h1 class="centered-title">CLIMATE ANALYTICS DASHBOARD</h1>
    <p class="subtitle">Explore Climate Trends Across Regions and Time Periods</p>
""", unsafe_allow_html=True)



# Custom background 
st.markdown("""
    <style>
    html, body, .main, .block-container, [data-testid="stAppViewContainer"], 
    [data-testid="stSidebar"], [class*="css"] {
        background-color: #d2e5e0 !important;
        font-family: 'Georgia', serif;
        color: black;
    }

    .stSelectbox label,
    .stRadio > label {
        color: #1a1a1a !important;
        font-weight: bold;
    }

    .stDataFrame, .stAltairChart {
        background-color: transparent !important;
        color: black !important;
    }
            
    /* Rounded corners + subtle shadows */
    .stDataFrame, .stAltairChart, .element-container {
        background-color: #fff0 !important;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 10px;
}
    </style>
<style>
/* Sidebar background and text */
[data-testid="stSidebar"] {
    background-color: #edd7ca !important;  /* slightly darker green */
    color: #111 !important;
    border-right: 2px solid #4a604a;
}

/* Sidebar text and headers */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span  {
    color: #111 !important;
    font-family: 'Georgia', serif;
}

/* Sidebar labels (radio/select/checkbox/etc.) */
[data-testid="stSidebar"] label {
    color: #f0f0f0 !important;
    font-weight: bold;
}

/* Sidebar hover effects */
[data-testid="stSidebar"] label:hover {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    cursor: pointer;
    padding-left: 4px;
}
</style>

""", unsafe_allow_html=True)


page = st.sidebar.radio("GO TO:", ["Explore", "Compare Regions", "Geospatial View"])

uploaded_file = st.sidebar.file_uploader("📂 Upload Climate CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.title()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
else:
    st.warning("⚠️ Please upload a dataset to continue.")
    st.stop()

if page == "Explore":
    show_dashboard(df)
elif page == "Compare Regions":
    show_compare(df)
elif page == "Geospatial View":
    show_map(df)
