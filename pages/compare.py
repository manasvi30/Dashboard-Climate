import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np

# -------------------- Page setup -------------------- #
st.set_page_config(page_title="Compare Districts", layout="wide")

# -------------------- Full Green Background -------------------- #
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
if mode == "Select District":
    st.switch_page("dash.py")

elif mode == "Map":
    st.switch_page("pages/map.py") 

#Load data
df = pd.read_csv("dailyclimate.csv", index_col=0)
df.columns = df.columns.str.strip()
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

numeric_columns = df.select_dtypes(include='number').columns.tolist()
if 'Unnamed: 0' in numeric_columns:
    numeric_columns.remove('Unnamed: 0')

st.markdown("### Two Districts Comparison")
# Options
col1, col2, col3, col4 = st.columns(4)
with col1:
    district1 = st.selectbox("🌍 Select First District", sorted(df['District'].unique()), key="district1")
with col2:
    district2 = st.selectbox("🌍 Select Second District", sorted(df['District'].unique()), key="district2")
with col3:
    x_axis = st.selectbox("📈 X-axis Column", df.columns)
with col4:
    y_axis = st.selectbox("📉 Y-axis Column (Numeric)", numeric_columns)

#frequency 
frequency = st.radio("📅 View Data By", ["Daily", "Monthly", "Yearly"], horizontal=True)

#Preprocessing Function 
def preprocess_data(district):
    filtered = df[df['District'] == district].copy()
    if frequency == "Monthly" and 'date' in filtered.columns:
        filtered['Month'] = filtered['date'].dt.to_period('M').astype(str)
        filtered = filtered.groupby(['Month']).mean(numeric_only=True).reset_index()
        time_col = 'Month'
    elif frequency == "Yearly" and 'date' in filtered.columns:
        filtered['Year'] = filtered['date'].dt.year
        filtered = filtered.groupby(['Year']).mean(numeric_only=True).reset_index()
        time_col = 'Year'
    else:
        time_col = x_axis
    return filtered, time_col

viz_df1, x1 = preprocess_data(district1)
viz_df2, x2 = preprocess_data(district2)

#Sample Data 
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 📍 {district1}")
    st.dataframe(viz_df1.head(), use_container_width=True)
with col2:
    st.markdown(f"### 📍 {district2}")
    st.dataframe(viz_df2.head(), use_container_width=True)

# -------------------- Line & Bar Charts -------------------- #
col1, col2 = st.columns(2)
with col1:
    st.line_chart(viz_df1.set_index(x1)[y_axis])
    st.bar_chart(viz_df1.set_index(x1)[y_axis])
with col2:
    st.line_chart(viz_df2.set_index(x2)[y_axis])
    st.bar_chart(viz_df2.set_index(x2)[y_axis])

# -------------------- Scatter Plot Comparison -------------------- #
st.markdown("---")
st.markdown("## 🔵 Combined Scatter Comparison")

viz_df1['District'] = district1
viz_df2['District'] = district2
combined_df = pd.concat([
    viz_df1[[x1, y_axis, 'District']].rename(columns={x1: 'Time'}),
    viz_df2[[x2, y_axis, 'District']].rename(columns={x2: 'Time'})
])
combined_df['Time'] = pd.to_datetime(combined_df['Time'], errors='coerce')

scatter_chart = alt.Chart(combined_df).mark_circle(size=60).encode(
    x='Time:T',
    y=alt.Y(y_axis, title=y_axis),
    color=alt.Color('District:N'),
    tooltip=['Time', y_axis, 'District']
).interactive().properties(height=400)
st.altair_chart(scatter_chart, use_container_width=True)

# -------------------- Correlation Matrix -------------------- #
st.markdown("---")
st.markdown("## 🔗 Correlation Between Weather Parameters")

col1, col2 = st.columns(2)
for i, (df_set, dist) in enumerate(zip([viz_df1, viz_df2], [district1, district2])):
    corr_matrix = df_set[numeric_columns].corr()
    corr_data = corr_matrix.reset_index().melt('index')
    corr_data.columns = ['Variable 1', 'Variable 2', 'Correlation']
    chart = alt.Chart(corr_data).mark_rect().encode(
        x=alt.X('Variable 2:O', sort=None),
        y=alt.Y('Variable 1:O', sort=None),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=(-1, 1))),
        tooltip=['Variable 1', 'Variable 2', 'Correlation']
    ).properties(title=f"{dist} Correlation Matrix", height=350)
    with (col1 if i == 0 else col2):
        st.altair_chart(chart, use_container_width=True)

# -------------------- Pairplot -------------------- #
st.markdown("---")
st.markdown("## 📉 Pairwise Relationships (Pairplot)")

selected_pairplot_cols = st.multiselect(
    "📌 Select Weather Parameters for Pairplot",
    numeric_columns,
    default=[y_axis],
    max_selections=6
)

if len(selected_pairplot_cols) >= 2:
    col1, col2 = st.columns(2)
    with st.spinner("Generating pairplots..."):
        with col1:
            fig1 = sns.pairplot(viz_df1[selected_pairplot_cols], corner=True)
            st.pyplot(fig1)
        with col2:
            fig2 = sns.pairplot(viz_df2[selected_pairplot_cols], corner=True)
            st.pyplot(fig2)

# -------------------- Clustering -------------------- #
st.markdown("---")
st.markdown("## 🧬 Weather Pattern Clustering")

clustering_mode = st.radio("Choose clustering mode:", [
    "Per District (Time-Series)", 
    "All Districts (Aggregate + Silhouette)"
])

# -------------------- Mode 1: Per District Clustering -------------------- #
if clustering_mode == "Per District (Time-Series)":
    cluster_features = st.multiselect(
        "📌 Select Features for Clustering",
        numeric_columns,
        default=[y_axis],
        key="clustering_features"
    )

    if len(cluster_features) >= 2:
        n_clusters = st.slider("🔢 Select Number of Clusters (K)", 2, 10, 3)

        for df_set, dist in zip([viz_df1, viz_df2], [district1, district2]):
            st.markdown(f"### 📍 {dist} Clustering")
            cluster_df = df_set[cluster_features].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_df)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)

            x_plot = st.selectbox("🧭 X-axis for Cluster Plot", cluster_features, index=0, key=f"x_{dist}")
            y_plot = st.selectbox("🧭 Y-axis for Cluster Plot", cluster_features, index=1 if len(cluster_features) > 1 else 0, key=f"y_{dist}")

            scatter_cluster = alt.Chart(cluster_df).mark_circle(size=60).encode(
                x=alt.X(x_plot, title=x_plot),
                y=alt.Y(y_plot, title=y_plot),
                color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
                tooltip=cluster_features + ['Cluster']
            ).interactive().properties(height=400)

            st.altair_chart(scatter_cluster, use_container_width=True)
    else:
        st.info("Please select at least two numeric features to perform clustering.")

# -------------------- Mode 2: All Districts + Silhouette -------------------- #
elif clustering_mode == "All Districts (Aggregate + Silhouette)":
    district_cluster_cols = st.multiselect(
        "📌 Select Features to Cluster All Districts",
        numeric_columns,
        key="district_silhouette"
    )

    if len(district_cluster_cols) >= 2:
        n_clusters = st.slider("🔢 Select Number of Clusters (K)", 2, 10, 3, key="silhouette_k")

        cluster_df = df[['District'] + district_cluster_cols].dropna()
        agg_df = cluster_df.groupby('District').mean().reset_index()

        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(agg_df[district_cluster_cols])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(scaled_df)
        agg_df['Cluster'] = labels

        # 📉 Silhouette Score
        silhouette_vals = silhouette_samples(scaled_df, labels)
        avg_score = silhouette_score(scaled_df, labels)
        st.markdown(f"**Average Silhouette Score:** `{avg_score:.3f}`")

        # 📊 Silhouette Plot
        fig, ax = plt.subplots(figsize=(7, 4))
        y_lower = 10

        for i in range(n_clusters):
            cluster_vals = silhouette_vals[labels == i]
            cluster_vals.sort()
            size = cluster_vals.shape[0]
            y_upper = y_lower + size

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size, str(i))
            y_lower = y_upper + 10

        ax.axvline(avg_score, color="red", linestyle="--")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title("Silhouette Plot for All-District K-Means Clustering")
        st.pyplot(fig)
    else:
        st.info("Please select at least two features to perform silhouette-based clustering.")

