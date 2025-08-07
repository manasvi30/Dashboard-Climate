

# Climate Dashboard – Exploratory Climate Data Analysis & Visualization

## Overview

This project presents an **interactive web-based dashboard** for exploring and visualizing meteorological datasets across districts and regions. Built with **Streamlit**, the dashboard supports time-series trend analysis, anomaly detection, clustering, feature distribution plots, and geospatial mapping — all through a user-friendly interface.

## Features

### Data Upload & Preprocessing
- Upload your own CSV files with region-wise time-series climate data.
- Cleans column names and parses date fields automatically.
- Supports daily, monthly, and yearly data aggregation.

### Single District Analysis
- Line and area charts for time-based trends.
- Box plots, histograms, and KDE plots for distribution analysis.
- Correlation matrices and pairplots to explore feature relationships.
- Clustering using KMeans with custom feature selection.
- Z-score based anomaly detection highlighted on time series.

### District Comparison
- Compare two districts side-by-side using various visualizations.
- Scatter plots, bar charts, and line charts for temporal comparison.
- Clustering comparison with silhouette score visualization.

### Geospatial Mapping
- Choropleth maps of Nepal using custom GeoJSON.
- Year-wise visualization of any numeric climate feature.
- Color scale selection and automatic annotation of max/min regions.
- Intelligent district name matching and validation.

---

## Datasets

- **Primary Dataset:**  
  [District-wise Monthly Climate Data for Nepal (1981–2019)](https://opendatanepal.com/dataset/district-wise-daily-climate-data-for-nepal)  
  > Includes over 14,000 records across 62 districts with parameters like:
  - Precipitation
  - Relative & Specific Humidity
  - Max/Min/Avg Temperature
  - Wind Speed
  - Wet Bulb Temp, Earth Skin Temp, Surface Pressure

- **Supplementary Source:**  
  [NASA POWER API](https://power.larc.nasa.gov/data-access-viewer/) used to fill missing data.

---

## Technologies Used

- **Frontend/UI:** Streamlit
- **Data Handling:** Pandas, NumPy
- **Visualization:** Plotly, Altair, Seaborn, Matplotlib
- **ML/Analytics:** Scikit-learn (KMeans, Z-score), GeoPandas
- **Geospatial:** [GeoJSON Nepal](https://github.com/mesaugat/geojson-nepal)

---


## Run Locally

```bash
git clone https://github.com/yourusername/climate-dashboard.git
cd climate-dashboard
pip install streamlit pandas numpy plotly altair scikit-learn matplotlib seaborn geopandas
streamlit run app.py
```

<h2> Climate Dashboard Screens</h2>
### First Page
   <img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Start%20Page.png?raw=true" width="380"/>
   
### Single District
<table>
  <tr>
     <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single2.png?raw=true" width="380"/></td>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single.png?raw=true" width="380"/></td>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single1.png?raw=true" width="380"/></td>
  </tr>

  <tr>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single3.png?raw=true" width="380"/></td>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single4.png?raw=true" width="380"/></td>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single5.png?raw=true" width="380"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Single6.png?raw=true" width="380"/></td>
  </tr>
</table>

### Compare District
<table>
  <tr>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Compare.png?raw=true" width="380"/></td>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Compare2.png?raw=true" width="380"/></td>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Compare3.png?raw=true" width="380"/></td>
  </tr>
</table>

### Map
<table>
  <tr>
    <td><img src="https://github.com/manasvi30/climate-dashboard/blob/dem/Images/Map.png?raw=true" width="380"/></td>
  </tr>
</table>





