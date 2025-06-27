# app.py
import streamlit as st
import pandas as pd
from dash import show_dashboard
from compare import show_compare
from map import show_map

st.set_page_config(page_title="CLIMATE DASHBOARD", layout="wide")
# Title ra Subtitle
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

# Background & Styling
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
            
    .stDataFrame, .stAltairChart, .element-container {
        background-color: #fff0 !important;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 10px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #edd7ca !important;
        color: #111 !important;
        border-right: 2px solid #4a604a;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #111 !important;
        font-family: 'Georgia', serif;
    }

    [data-testid="stSidebar"] label {
        color: #f0f0f0 !important;
        font-weight: bold;
    }

    [data-testid="stSidebar"] label:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        cursor: pointer;
        padding-left: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("GO TO:", ["Explore", "Compare Regions", "Geospatial View"])

# Multiple CSV
uploaded_files = st.sidebar.file_uploader("📂 Upload One or More Climate CSVs", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        df_temp = pd.read_csv(file)
        df_temp.columns = df_temp.columns.str.strip().str.title()
        if 'Date' in df_temp.columns:
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df.dropna(how='all', inplace=True)

else:
    st.warning("⚠️ Please upload at least one dataset to continue.")
    st.stop()

# Route to pages
if page == "Explore":
    show_dashboard(df)
elif page == "Compare Regions":
    show_compare(df)
elif page == "Geospatial View":
    show_map(df)
