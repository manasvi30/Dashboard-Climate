import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

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

#drop down feature
col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    district = st.selectbox("🌍 Select District", sorted(df['District'].unique()))
with col2:
    x_axis = st.selectbox("📈 X-axis Column", available_columns)
with col3:
    y_axis = st.selectbox("📉 Y-axis Column", numeric_columns)

# filter
filtered_df = df[df['District'] == district]
st.subheader(f"{district}")

#radio button 
frequency = st.radio("📅 View Data By", ["Daily", "Monthly", "Yearly"], horizontal=True)

# aggregate data according to frequency
if frequency == "Monthly" and 'date' in filtered_df.columns:
    viz_df = filtered_df.copy()
    viz_df['Month'] = viz_df['date'].dt.to_period('M').astype(str)
    viz_df = viz_df.groupby(['Month']).mean(numeric_only=True).reset_index()
    x_axis_used = 'Month'
elif frequency == "Yearly" and 'date' in filtered_df.columns:
    viz_df = filtered_df.copy()
    viz_df['Year'] = viz_df['date'].dt.year
    viz_df = viz_df.groupby(['Year']).mean(numeric_only=True).reset_index()
    x_axis_used = 'Year'
else:
    # Daily 
    viz_df = filtered_df.copy()
    x_axis_used = x_axis

#sample data 
st.markdown("### 📋 Sample Data")
st.dataframe(viz_df.head(), use_container_width=True)

#feature statistics
st.markdown("### 📊 Feature Statistics")
st.dataframe(viz_df.select_dtypes(include='number').describe().T, use_container_width=True)

#prepare data for chart 
plot_data = viz_df[[x_axis_used, y_axis]].dropna()
x_data = plot_data[x_axis_used]
plot_data[x_axis_used] = pd.to_datetime(x_data) if pd.api.types.is_datetime64_any_dtype(x_data) else x_data.astype(str)

#row: line ra bar
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"#### 📈 Line Chart: {y_axis} vs {x_axis_used}")
    st.line_chart(plot_data.set_index(x_axis_used))
with col2:
    st.markdown(f"#### 📊 Bar Chart: Average {y_axis} per {x_axis_used}")
    bar_data = plot_data.groupby(x_axis_used)[y_axis].mean().sort_index()
    st.bar_chart(bar_data)

#row: scatter ra box 
st.markdown("---") 
col3, col4 = st.columns(2)
with col3:
    st.markdown(f"#### 🔵 Scatter Plot: {y_axis} vs {x_axis_used}")
    scatter_chart = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X(x_axis_used, title=x_axis_used),
        y=alt.Y(y_axis, title=y_axis),
        tooltip=[x_axis_used, y_axis]
    ).interactive().properties(height=300)
    st.altair_chart(scatter_chart, use_container_width=True)
with col4:
    st.markdown(f"#### 📦 Box Plot: {y_axis} Distribution")
    box_chart = alt.Chart(plot_data).mark_boxplot(extent='min-max').encode(
        y=alt.Y(y_axis, title=y_axis),
        tooltip=[y_axis]
    ).properties(height=300)
    st.altair_chart(box_chart, use_container_width=True)

# visualizations
st.markdown("---")
st.markdown("## 🌦️ Visualization of Weather Parameters and Distributions")

default_params = [y_axis]
if x_axis in numeric_columns and x_axis != y_axis:
    default_params.append(x_axis)

selected_params = st.multiselect("📌 Select Weather Parameters to Visualize", numeric_columns, default=default_params)

if selected_params:
    for param in selected_params:
        st.markdown(f"### 📊 Distribution of {param}")
        col1, col2 = st.columns(2)
        with col1:
            hist = alt.Chart(viz_df).mark_bar(opacity=0.7, color="#54717A").encode(
                alt.X(param, bin=alt.Bin(maxbins=30), title=f"{param} (Binned)"),
                y='count()',
                tooltip=[param]
            ).properties(height=250)
            st.altair_chart(hist, use_container_width=True)
        with col2:
            kde = alt.Chart(viz_df).transform_density(
                param,
                as_=[param, 'density'],
            ).mark_area(opacity=0.4, color="#9BDEAC").encode(
                x=alt.X(param, title=param),
                y='density:Q'
            ).properties(height=250)
            st.altair_chart(kde, use_container_width=True)

#correlation matrix 
st.markdown("---")
st.markdown("## 🔗 Correlation Between Weather Parameters")

corr_matrix = viz_df[numeric_columns].corr()
corr_data = corr_matrix.reset_index().melt('index')
corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']

heatmap = alt.Chart(corr_data).mark_rect().encode(
    x=alt.X('Variable 2:O', sort=None, title=None),
    y=alt.Y('Variable 1:O', sort=None, title=None),
    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
    tooltip=['Variable 1', 'Variable 2', 'Correlation']
).properties(height=400)
st.altair_chart(heatmap, use_container_width=True)

#pairplot 
st.markdown("---")
st.markdown("## 📉 Pairwise Relationships (Pairplot)")

default_pairplot = [y_axis]
if x_axis in numeric_columns and x_axis != y_axis:
    default_pairplot.append(x_axis)

selected_pairplot_cols = st.multiselect(
    "📌 Select Weather Parameters for Pairplot (Max 10 for clarity)",
    numeric_columns,
    default=default_pairplot,
    max_selections=10
)

if len(selected_pairplot_cols) >= 2:
    with st.spinner("Generating pairplot..."):
        pairplot_fig = sns.pairplot(viz_df[selected_pairplot_cols], height=2.2, corner=True)
        st.pyplot(pairplot_fig)
else:
    st.info("Please select at least 2 parameters for the pairplot.")

#clustering
st.markdown("---")
st.markdown("## 🧬 Weather Pattern Clustering")

cluster_features = st.multiselect(
    "📌 Select Features for Clustering",
    numeric_columns,
    default=default_pairplot
)

if len(cluster_features) >= 2:
    n_clusters = st.slider("🔢 Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

    cluster_df = viz_df[cluster_features].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(scaled_data)

    cluster_df = cluster_df.copy()
    cluster_df['Cluster'] = cluster_labels

    st.markdown("### 📍 Clustered Scatter Plot")
    x_plot = st.selectbox("🧭 X-axis for Cluster Plot", cluster_features, index=0)
    y_plot = st.selectbox("🧭 Y-axis for Cluster Plot", cluster_features, index=1 if len(cluster_features) > 1 else 0)

    scatter_cluster = alt.Chart(cluster_df).mark_circle(size=60).encode(
        x=alt.X(x_plot, title=x_plot),
        y=alt.Y(y_plot, title=y_plot),
        color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
        tooltip=cluster_features + ['Cluster']
    ).interactive().properties(height=400)

    st.altair_chart(scatter_cluster, use_container_width=True)
else:
    st.info("Please select at least two numeric features to perform clustering.")
