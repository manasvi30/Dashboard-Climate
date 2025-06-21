import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Nepal's Climate Dashboard", layout="wide")

#background green parna
st.markdown("""
    <style>
    html, body, .main, .block-container, [data-testid="stAppViewContainer"], 
    [data-testid="stSidebar"], [class*="css"] {
        background-color: #799F7F !important;
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
    </style>
""", unsafe_allow_html=True)

st.title("Nepal's Climate Dashboard")

#radio button
mode = st.radio("View Mode", ["Single District", "Compare District", "Map"], horizontal=True)

#switch page
if mode == "Compare District":
    st.switch_page("pages/compare.py")

elif mode == "Map":
    st.switch_page("pages/map.py") 

#data
df = pd.read_csv("dailyclimate.csv", index_col=0)
df.columns = df.columns.str.strip()
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

available_columns = [col for col in df.columns if col != 'Unnamed: 0']
numeric_columns = df.select_dtypes(include='number').columns.tolist()
if 'Unnamed: 0' in numeric_columns:
    numeric_columns.remove('Unnamed: 0')