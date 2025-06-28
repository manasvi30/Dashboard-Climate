import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def show_dashboard(df):
    st.header("\U0001F4CA District-wise Yearly Climate")

    available_columns = [col for col in df.columns if col != 'Unnamed: 0']
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if 'Unnamed: 0' in numeric_columns:
        numeric_columns.remove('Unnamed: 0')

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        district = st.selectbox("Select District", sorted(df['District'].unique()))
    with col2:
        x_axis = st.selectbox("X-axis Column", available_columns)
    with col3:
        y_axis = st.selectbox("Y-axis Column", numeric_columns)

    filtered_df = df[df['District'] == district]
    st.subheader(f"{district}")

    viz_df = filtered_df.copy()
    x_axis_used = x_axis

    # Pairplot define
    default_pairplot = [y_axis]
    if x_axis in numeric_columns and x_axis != y_axis:
        default_pairplot.append(x_axis)

    tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "Advanced Analysis"])

    with tab1:
        plot_data = viz_df[[x_axis_used, y_axis]].dropna()
        x_data = plot_data[x_axis_used]
        plot_data[x_axis_used] = pd.to_datetime(x_data) if pd.api.types.is_datetime64_any_dtype(x_data) else x_data.astype(str)

        if pd.api.types.is_datetime64_any_dtype(plot_data[x_axis_used]):
            start_date, end_date = st.date_input("\U0001F4C6 Filter by Date Range", [plot_data[x_axis_used].min(), plot_data[x_axis_used].max()])
            plot_data = plot_data[(plot_data[x_axis_used] >= pd.to_datetime(start_date)) & (plot_data[x_axis_used] <= pd.to_datetime(end_date))]

        st.markdown("### \U0001F4CB Sample Data")
        st.dataframe(viz_df.head(), use_container_width=True)

        st.markdown("### \U0001F4CA Feature Statistics")
        st.dataframe(viz_df.select_dtypes(include='number').describe().T, use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"#### \U0001F535 Scatter Plot: {y_axis} vs {x_axis_used}")
            scatter_chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X(x_axis_used, title=x_axis_used),
                y=alt.Y(y_axis, title=y_axis),
                tooltip=[x_axis_used, y_axis]
            ).interactive().properties(height=300)
            st.altair_chart(scatter_chart, use_container_width=True)

        with col4:
            st.markdown(f"#### \U0001F4E6 Box Plot: {y_axis} Distribution")
            box_chart = alt.Chart(plot_data).mark_boxplot(extent='min-max').encode(
                y=alt.Y(y_axis, title=y_axis),
                tooltip=[y_axis]
            ).properties(height=300)
            st.altair_chart(box_chart, use_container_width=True)

        st.markdown(f"#### 📊 Bar Chart: Average {y_axis} per {x_axis_used}")
        bar_data = plot_data.groupby(x_axis_used)[y_axis].mean().sort_index().reset_index()
        fig = px.bar(bar_data, x=x_axis_used, y=y_axis)
        st.plotly_chart(fig, use_container_width=False)


    with tab2:
        st.markdown(f"#### \U0001F4C8 Line or Area Chart {y_axis} vs {x_axis_used}")
        with st.expander("\U0001F4C8 Optional: Add Rolling Average Line"):
            window = st.slider("Rolling Window (days)", 1, 60, 7)
            plot_data[f'{y_axis}_Smoothed'] = plot_data[y_axis].rolling(window=window).mean()

        chart_type = st.radio("Chart Type", ["Line Chart", "Area Chart"], horizontal=True)
        chart_data = plot_data.set_index(x_axis_used)[[y_axis, f'{y_axis}_Smoothed']]

        if chart_type == "Line Chart":
            st.line_chart(chart_data)
        else:
            st.area_chart(chart_data)
        st.markdown("### \U0001F4C5 Export Filtered Data from line or area chart")
        csv = plot_data.to_csv(index=False).encode('utf-8')
        st.download_button("\u2B07\uFE0F Download CSV", csv, "climate_data.csv", "text/csv")
        
        st.markdown("### \U0001F326\uFE0F Distribution of Weather Parameters")
        default_params = [y_axis]
        if x_axis in numeric_columns and x_axis != y_axis:
            default_params.append(x_axis)

        selected_params = st.multiselect("Select Weather Parameters to Visualize", numeric_columns, default=default_params)
        if selected_params:
            for param in selected_params:
                st.markdown(f"### \U0001F4CA Distribution of {param}")
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
        st.markdown("### 📉 Pairwise Relationships (Pairplot)")
        default_pairplot = [y_axis]
        if x_axis in numeric_columns and x_axis != y_axis:
            default_pairplot.append(x_axis)

        selected_pairplot_cols = st.multiselect(
            "Select Weather Parameters for Pairplot (Max 10 for clarity)",
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

        st.markdown("### \U0001F517 Correlation Between Weather Parameters")
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

        if x_axis in corr_matrix.columns and y_axis in corr_matrix.columns:
            corr_value = corr_matrix.loc[x_axis, y_axis]
            if abs(corr_value) > 0.7:
                st.success(f"\u2705 Strong correlation ({corr_value:.2f}) between {x_axis} and {y_axis}.")
            elif abs(corr_value) > 0.4:
                st.warning(f"\u26A0\uFE0F Moderate correlation ({corr_value:.2f}) detected.")
            else:
                st.info(f"\u2139\uFE0F Weak or no correlation ({corr_value:.2f}).")

    with tab3:
        st.markdown("### \U0001F9EC Weather Pattern Clustering")
        cluster_features = st.multiselect(
            "Select Features for Clustering",
            numeric_columns,
            default=default_pairplot
        )

        if len(cluster_features) >= 2:
            n_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=3)

            cluster_df = viz_df[cluster_features].dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_df)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(scaled_data)

            cluster_df = cluster_df.copy()
            cluster_df['Cluster'] = cluster_labels

            x_plot = st.selectbox("X-axis for Cluster Plot", cluster_features, index=0)
            y_plot = st.selectbox("Y-axis for Cluster Plot", cluster_features, index=1 if len(cluster_features) > 1 else 0)

            scatter_cluster = alt.Chart(cluster_df).mark_circle(size=60).encode(
                x=alt.X(x_plot, title=x_plot),
                y=alt.Y(y_plot, title=y_plot),
                color=alt.Color('Cluster:N', legend=alt.Legend(title="Cluster")),
                tooltip=cluster_features + ['Cluster']
            ).interactive().properties(height=400)

            st.altair_chart(scatter_cluster, use_container_width=True)

        else:
            st.info("Please select at least two numeric features to perform clustering.")

        st.markdown("### \U0001F6A8 Outlier / Anomaly Detection")
        outlier_col = st.selectbox("Select Variable to Detect Anomalies", numeric_columns, key="outlier_var")

        filtered_df = df[df['District'] == district].copy()
        filtered_df["Z_Score"] = (filtered_df[outlier_col] - filtered_df[outlier_col].mean()) / filtered_df[outlier_col].std()
        threshold = 2
        filtered_df["Anomaly"] = filtered_df["Z_Score"].abs() > threshold

        fig = px.line(filtered_df, x="Date", y=outlier_col, title=f"{outlier_col} Over Time in {district}")
        fig.add_scatter(
            x=filtered_df[filtered_df["Anomaly"]]["Date"],
            y=filtered_df[filtered_df["Anomaly"]][outlier_col],
            mode='markers',
            marker=dict(color='red', size=10),
            name="Anomalies"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### \U0001F9FE Anomalies Detected")
        st.dataframe(filtered_df[filtered_df["Anomaly"]][["Date", outlier_col, "Z_Score"]])
