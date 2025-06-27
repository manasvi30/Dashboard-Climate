import streamlit as st
import pandas as pd
import plotly.express as px
import requests

def show_map(df):
    st.header("🗺️ Climate Choropleth Map")

    # ✅ Notify user about map limitation
    st.info("🌐 This choropleth map is designed for datasets containing districts from Nepal.")

    if 'District' not in df.columns:
        st.error("❌ Your dataset must contain a 'District' column.")
        return

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        st.warning("⚠️ No numeric data found to map.")
        return

    selected_feature = st.selectbox("🌡️ Select a Climate Feature", numeric_columns)

    # 📦 Load Nepal GeoJSON from GitHub
    geo_url = "https://raw.githubusercontent.com/mesaugat/geojson-nepal/master/nepal-districts.geojson"
    try:
        geo_data = requests.get(geo_url).json()
    except Exception as e:
        st.error("❌ Failed to load GeoJSON data.")
        return

    # 🧼 Normalize district names for matching
    df_map = df.copy()
    df_map['District'] = df_map['District'].str.strip().str.upper()
    district_fixes = {
        "Bajang": "Bajhang",
        "Chitawan": "Chitwan",
        "Dolkha": "Dolakha",
        "Kabhre": "Kavrepalanchok",
        "Panchther": "Panchthar",
        "Routahat": "Rautahat"
    }
    df_map['District'] = df_map['District'].replace(district_fixes)
    df_map = df_map.groupby("District")[selected_feature].mean().reset_index()

    # 🧩 Known Nepal districts from GeoJSON
    nepal_geo_districts = [feature['properties']['DISTRICT'].strip().upper() for feature in geo_data['features']]
    matched_districts = df_map['District'].isin(nepal_geo_districts)
    unmatched_count = (~matched_districts).sum()

    if unmatched_count > 0:
        unmatched = df_map[~matched_districts]['District'].tolist()
        st.text(f"Unmatched Districts: {', '.join(unmatched)}")

    df_map = df_map[df_map['District'].isin(nepal_geo_districts)]

    # 🗺️ Build the map
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
