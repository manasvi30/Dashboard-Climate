import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def show_compare(df):
    st.header("📊 Compare Districts")

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if 'Unnamed: 0' in numeric_columns:
        numeric_columns.remove('Unnamed: 0')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        district1 = st.selectbox("Select First District", sorted(df['District'].unique()), key="district1")
    with col2:
        district2 = st.selectbox("Select Second District", sorted(df['District'].unique()), key="district2")
    with col3:
        x_axis = st.selectbox("X-axis Column", df.columns)
    with col4:
        y_axis = st.selectbox("Y-axis Column (Numeric)", numeric_columns)

    tab1, tab2, tab3 = st.tabs(["📈 Visual Comparison", "📊 Statistical Comparison", "🧠 Clustering"])

    with tab1:
        def preprocess_data(district):
            filtered = df[df['District'] == district].copy()
            return filtered, x_axis

        viz_df1, x1 = preprocess_data(district1)
        viz_df2, x2 = preprocess_data(district2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"###  {district1}")
            st.dataframe(viz_df1.head(), use_container_width=True)

            st.markdown(f"### 📈 Bar Chart: Average {y_axis} per {x_axis}")
            bar_data_1 = viz_df1.groupby(x1)[y_axis].mean().sort_index()
            st.bar_chart(bar_data_1)
           
            st.markdown(f"### 📈 Line or Area Chart: {y_axis} vs {x1}")
            chart_type1 = st.radio(f"Chart Type for {district1}", ["Line Chart", "Area Chart"], horizontal=True, key="chart_type1")
            # Use only raw data
            chart_data1 = viz_df1.set_index(x1)[[y_axis]]

            if chart_type1 == "Line Chart":
                st.line_chart(chart_data1)
            else:
                st.area_chart(chart_data1)


        with col2:
            st.markdown(f"###  {district2}")
            st.dataframe(viz_df2.head(), use_container_width=True)

            st.markdown(f"### 📈 Bar Chart: Average {y_axis} per {x_axis}")
            bar_data_2 = viz_df2.groupby(x2)[y_axis].mean().sort_index()
            st.bar_chart(bar_data_2)

            st.markdown(f"### 📈 Line or Area Chart for {district2}: {y_axis} vs {x2}")
            chart_type2 = st.radio(
            f"Chart Type for {district2}", 
            ["Line Chart", "Area Chart"], 
            horizontal=True, 
            key=f"chart_type_{district2}"
            )
            chart_data2 = viz_df2.set_index(x2)[[y_axis]]
            if chart_type2 == "Line Chart":
                st.line_chart(chart_data2)
            else:
                st.area_chart(chart_data2)

    with tab2:
        st.markdown("## 🔹 Combined Scatter Comparison")
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

        st.markdown("## 📉 Pairwise Relationships (Pairplot)")
        selected_pairplot_cols = st.multiselect(
            "Select Weather Parameters for Pairplot",
            numeric_columns,
            default=[y_axis],
            max_selections=6
        )
        if len(selected_pairplot_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = sns.pairplot(viz_df1[selected_pairplot_cols], corner=True)
                st.pyplot(fig1)
            with col2:
                fig2 = sns.pairplot(viz_df2[selected_pairplot_cols], corner=True)
                st.pyplot(fig2)

    with tab3:
        st.markdown("## 🧠 Clustering Analysis")
        cluster_features = st.multiselect(
            "Select Features for Clustering",
            numeric_columns,
            default=[y_axis],
            key="clustering_features"
        )

        if len(cluster_features) >= 2:
            x_plot = st.selectbox("X-axis for Cluster Plot", cluster_features, index=0, key="shared_x")
            y_plot = st.selectbox("Y-axis for Cluster Plot", cluster_features, index=1 if len(cluster_features) > 1 else 0, key="shared_y")

            clustering_mode = st.radio("Choose clustering mode:", [
                "Per District (Time-Series)", 
                "All Districts (Aggregate + Silhouette)"
            ])

            if clustering_mode == "Per District (Time-Series)":
                n_clusters = st.slider("Select Number of Clusters (K)", 2, 10, 3)
                for df_set, dist in zip([viz_df1, viz_df2], [district1, district2]):
                    st.markdown(f"### 📍 {dist} Clustering")
                    cluster_df = df_set[cluster_features].dropna()
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_df)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                    cluster_df['Cluster'] = kmeans.fit_predict(scaled_data)

                    scatter_cluster = alt.Chart(cluster_df).mark_circle(size=60).encode(
                        x=alt.X(x_plot, title=x_plot),
                        y=alt.Y(y_plot, title=y_plot),
                        color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
                        tooltip=cluster_features + ['Cluster']
                    ).interactive().properties(height=400)
                    st.altair_chart(scatter_cluster, use_container_width=True)

            elif clustering_mode == "All Districts (Aggregate + Silhouette)":
                n_clusters = st.slider("Select Number of Clusters (K)", 2, 10, 3, key="silhouette_k")
                cluster_df = df[['District'] + cluster_features].dropna()
                agg_df = cluster_df.groupby('District').mean().reset_index()
                scaler = StandardScaler()
                scaled_df = scaler.fit_transform(agg_df[cluster_features])
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(scaled_df)
                agg_df['Cluster'] = labels
                silhouette_vals = silhouette_samples(scaled_df, labels)
                avg_score = silhouette_score(scaled_df, labels)
                st.markdown(f"**Average Silhouette Score:** `{avg_score:.3f}`")

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

