import streamlit as st
import pandas as pd
import plotly.express as px
import requests

def show_map(df):
    st.header("🗺️ Climate Choropleth Map")

    st.info("🌐 This choropleth map is designed for datasets containing districts from Nepal.")
    

    if 'District' not in df.columns:
        st.error("❌ Your dataset must contain a 'District' column.")
        return

    # Detect numeric columns for selection
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if 'Unnamed: 0' in numeric_columns:
        numeric_columns.remove('Unnamed: 0')

    if not numeric_columns:
        st.warning("⚠️ No numeric data found to map.")
        return

    selected_feature = st.selectbox("🌡️ Select a Climate Feature", numeric_columns)

    # Load Nepal district geojson
    geo_url = "https://raw.githubusercontent.com/mesaugat/geojson-nepal/master/nepal-districts.geojson"
    try:
        geo_data = requests.get(geo_url).json()
    except Exception as e:
        st.error("❌ Failed to load GeoJSON data.")
        return

    # Normalize district names in data
    df_map = df.copy()
    df_map['District'] = df_map['District'].astype(str).str.strip().str.upper()
    df_map = df_map.groupby("District")[selected_feature].mean().reset_index()

    # Districts from GeoJSON uppercas
    nepal_geo_districts = [feature['properties']['DISTRICT'].strip().upper() for feature in geo_data['features']]
    matched_districts = df_map['District'].isin(nepal_geo_districts)
    unmatched_count = (~matched_districts).sum()

    # Warn about mismatches
    if unmatched_count > 0:
        unmatched = df_map[~matched_districts]['District'].tolist()
        st.warning(f"⚠️ Unmatched Districts (not shown on map): {', '.join(unmatched)}")

    # Filter only matched districts
    df_map = df_map[matched_districts]

    # Draw map
    if not df_map.empty:
        fig = px.choropleth(
            df_map,
            geojson=geo_data,
            locations="District",
            featureidkey="properties.DISTRICT",
            color=selected_feature,
            color_continuous_scale="YlGnBu",
            projection="mercator",
            hover_name="District"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No valid districts found for choropleth visualization.")
