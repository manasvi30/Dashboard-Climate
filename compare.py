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
    st.markdown(
        """
        <h2>
            <img src="https://cdn-icons-png.flaticon.com/512/7756/7756168.png" width="30" style="vertical-align: middle; margin-right: 8px;">
            Compare Districts
        </h2>
        """,
        unsafe_allow_html=True
    )

    df.columns = df.columns.str.strip()

    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
    if not date_col:
        st.error("❌ No column named 'date' found. Please check your file.")
        return
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    start_date, end_date = st.date_input(
        "📆 Select Date Range for Analysis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]
    st.caption(f"Data filtered from **{start_date}** to **{end_date}**")

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if 'Unnamed: 0' in numeric_columns:
        numeric_columns.remove('Unnamed: 0')

    freq_option = st.radio("Choose Data Frequency", ["Daily", "Monthly", "Yearly"], horizontal=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        district1 = st.selectbox("Select First District", sorted(df['District'].unique()), key="district1")
    with col2:
        district2 = st.selectbox("Select Second District", sorted(df['District'].unique()), key="district2")
    with col3:
        x_axis = st.selectbox("X-axis Column", [col for col in df.columns if col != 'Unnamed: 0'])
    with col4:
        y_axis = st.selectbox("Y-axis Column", numeric_columns)

    def preprocess_data(district):
        filtered_df = df[df['District'] == district].copy()
        if freq_option == "Monthly":
            filtered_df['Period'] = filtered_df[date_col].dt.to_period('M').astype(str)
        elif freq_option == "Yearly":
            filtered_df['Period'] = filtered_df[date_col].dt.year.astype(str)
        else:
            filtered_df['Period'] = filtered_df[date_col].dt.date.astype(str)
        return filtered_df, 'Period'

    try:
        viz_df1, x1 = preprocess_data(district1)
        viz_df2, x2 = preprocess_data(district2)
    except Exception as e:
        st.error(f"⚠️ Failed during preprocessing: {e}")
        return

    tab1, tab2 = st.tabs([
        " Visual Comparison","Scatterplot and Clustering"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        for i, (viz_df, district, chart_key) in enumerate(zip([viz_df1, viz_df2], [district1, district2], ["chart_type1", "chart_type2"])):
            with [col1, col2][i]:

                st.markdown(
                    f"""
                    <h4>
                        <img src="https://cdn-icons-png.flaticon.com/512/8451/8451381.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                        {district} ({freq_option} view)
                    </h4>
                    """,
                    unsafe_allow_html=True
                )

                st.dataframe(viz_df.head(), use_container_width=True)

                st.markdown(
                    f"""
                    <h5>
                        <img src="https://cdn-icons-png.flaticon.com/512/4926/4926731.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                        {freq_option} Aggregated Data
                    </h5>
                    """,
                    unsafe_allow_html=True
                )
                aggregated_df = viz_df.groupby('Period')[numeric_columns].mean().reset_index()
                st.dataframe(aggregated_df, use_container_width=True)

                st.markdown(
                    f"""
                    <h5>
                        <img src="https://cdn-icons-png.flaticon.com/512/3586/3586022.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                        Bar Chart: Average {y_axis} per {x_axis}
                    </h5>
                    """,
                    unsafe_allow_html=True
                )
                if x_axis in aggregated_df.columns and y_axis in aggregated_df.columns:
                    plot_data = aggregated_df[[x_axis, y_axis]].dropna()
                    import plotly.express as px
                    fig = px.bar(plot_data, x=x_axis, y=y_axis)
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{district}")
                else:
                    st.warning(f"⚠️ {x_axis} or {y_axis} not found in aggregated data.")

                if freq_option == "Monthly":
                    viz_df['Period'] = pd.to_datetime(viz_df['Period'] + "-01", errors='coerce')
                elif freq_option == "Yearly":
                    viz_df['Period'] = pd.to_datetime(viz_df['Period'] + "-01-01", errors='coerce')
                else:
                    viz_df['Period'] = pd.to_datetime(viz_df['Period'], errors='coerce')

                agg_data = viz_df.groupby('Period', as_index=False)[y_axis].mean()

                st.markdown(
                    f"""
                    <h5>
                        <img src="https://cdn-icons-png.flaticon.com/512/7495/7495244.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                        Line or Area Chart: {y_axis} vs Period
                    </h5>
                    """,
                    unsafe_allow_html=True
                )
                chart_type = st.radio(f"Chart Type for {district}", ["Line Chart", "Area Chart"], horizontal=True, key=chart_key)

                if chart_type == "Line Chart":
                    chart = alt.Chart(agg_data).mark_line().encode(
                        x=alt.X("Period:T", title="Time"),
                        y=alt.Y(y_axis, title=y_axis),
                        tooltip=["Period", y_axis]
                    ).interactive()
                else:
                    chart = alt.Chart(agg_data).mark_area(opacity=0.5).encode(
                        x=alt.X("Period:T", title="Time"),
                        y=alt.Y(y_axis, title=y_axis),
                        tooltip=["Period", y_axis]
                    ).interactive()

                st.altair_chart(chart, use_container_width=True)

    with tab2:
        st.markdown(
            """
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/7837/7837488.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                Combined Scatter Comparison
            </h4>
            """,
            unsafe_allow_html=True
        )
        agg1 = viz_df1.groupby('Period')[[x_axis, y_axis]].mean().reset_index()
        agg1['District'] = district1
        agg2 = viz_df2.groupby('Period')[[x_axis, y_axis]].mean().reset_index()
        agg2['District'] = district2

        combined_agg_df = pd.concat([agg1, agg2], ignore_index=True).dropna()

        if not combined_agg_df.empty:
            scatter_chart_agg = alt.Chart(combined_agg_df).mark_circle(size=60).encode(
                x=alt.X(x_axis, title=x_axis),
                y=alt.Y(y_axis, title=y_axis),
                color=alt.Color('District:N', legend=alt.Legend(title="District")),
                tooltip=['Period', x_axis, y_axis, 'District']
            ).interactive().properties(height=400)
            st.altair_chart(scatter_chart_agg, use_container_width=True)
        else:
            st.warning("Not enough data to display the aggregated scatter plot.")

        st.markdown(
            """
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/5464/5464694.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Clustering Analysis
            </h4>
            """,
            unsafe_allow_html=True
        )
        cluster_features = st.multiselect(
            "Select Features for Clustering",
            numeric_columns,
            default=[y_axis],
            key="clustering_features"
        )

        if len(cluster_features) >= 2:
            x_plot = st.selectbox("X-axis for Cluster Plot", cluster_features, index=0, key="shared_x")
            y_plot = st.selectbox("Y-axis for Cluster Plot", cluster_features, index=1 if len(cluster_features) > 1 else 0, key="shared_y")

            clustering_mode = st.radio("Choose clustering mode:", ["Per District (Time-Series)", "All Districts (Aggregate + Silhouette)"])

            if clustering_mode == "Per District (Time-Series)":
                n_clusters = st.slider("Select Number of Clusters (K)", 2, 10, 3)
                for df_set, dist in zip([viz_df1, viz_df2], [district1, district2]):
                    st.markdown(f"### \U0001F4CD {dist} Clustering")
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
