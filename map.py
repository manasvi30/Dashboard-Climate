import streamlit as st
import pandas as pd
import plotly.express as px
import requests

def show_map(df):
    st.markdown(
        """
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/512/2969/2969964.png" width="40" style="vertical-align: middle; margin-right: 8px;">Climate Choropleth Map
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.info("This choropleth map is designed for datasets containing districts from Nepal.")

    if 'District' not in df.columns or 'Date' not in df.columns:
        st.error("Dataset must contain both 'District' and 'Date' columns.")
        return

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if 'Unnamed: 0' in numeric_columns:
        numeric_columns.remove('Unnamed: 0')
    if not numeric_columns:
        st.warning("No numeric data found to map.")
        return

    color_scales = px.colors.named_colorscales()
    default_scale = "ylgnbu" if "ylgnbu" in color_scales else color_scales[0]

    col1, col2 = st.columns(2)
    with col1:
        selected_feature = st.selectbox("Select a Climate Feature", numeric_columns)
    with col2:
        color_scale = st.selectbox("Select Color Scale", color_scales, index=color_scales.index(default_scale))

    min_year = df['Date'].dt.year.min()
    max_year = df['Date'].dt.year.max()
    selected_year = st.slider("Select Year", min_year, max_year, max_year)

    df_year = df[df['Date'].dt.year == selected_year]

    # Load GeoJSON from Mesaugat repo (legacy 77-districts)
    geo_url = "https://raw.githubusercontent.com/mesaugat/geojson-nepal/master/nepal-districts.geojson"
    try:
        geo_data = requests.get(geo_url).json()
    except Exception:
        st.error("Failed to load GeoJSON data.")
        return

    df_year['District'] = df_year['District'].astype(str).str.strip().str.upper()
    df_map = df_year.groupby("District")[selected_feature].mean().reset_index()

    geo_districts = [f['properties']['DISTRICT'].strip().upper() for f in geo_data['features']]
    matched_districts = df_map['District'].isin(geo_districts)
    unmatched_count = (~matched_districts).sum()

    if unmatched_count > 0:
        unmatched = df_map[~matched_districts]['District'].tolist()
        st.warning(f"Unmatched Districts (not shown): {', '.join(unmatched)}")

    df_map = df_map[matched_districts]
    if df_map.empty:
        st.error("No valid districts found for mapping.")
        return

    max_row = df_map.loc[df_map[selected_feature].idxmax()]
    min_row = df_map.loc[df_map[selected_feature].idxmin()]

    fig = px.choropleth(
        df_map,
        geojson=geo_data,
        locations="District",
        featureidkey="properties.DISTRICT",
        color=selected_feature,
        color_continuous_scale=color_scale,
        projection="mercator",
        hover_name="District"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    fig.add_annotation(
        text=f"<b>Highest</b>: {max_row['District'].title()}<br><b>Lowest</b>: {min_row['District'].title()}",
        x=1.02, y=0.2,
        xref="paper", yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=10, color="gray"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="lightgray",
        borderwidth=1
    )

    st.plotly_chart(fig, use_container_width=True)

# For testing - load your CSV file here
def main():
    st.title("🗺️ Nepal Climate Choropleth Demo")
    uploaded_file = st.file_uploader("Upload your dataset (with 'District' and 'Date')", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        show_map(df)
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()
