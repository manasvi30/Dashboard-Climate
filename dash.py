import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def show_dashboard(df):
    st.markdown(
        """
        <div style="display: flex; align-items: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/7756/7756168.png" width="24" style="margin-right: 8px;">
            <h2 style="margin: 0;">District-wise Climate Overview</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    df.columns = df.columns.str.strip()
    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
    if not date_col:
        st.error(" No column named 'date' found. Please check your file.")
        return
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Date Range 
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    start_date, end_date = st.date_input(
        "Select Date Range for Analysis",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]
    st.caption(f"Data filtered from **{start_date}** to **{end_date}**")

    freq_option = st.radio("Choose Data Frequency", ["Daily", "Monthly", "Yearly"], horizontal=True)

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

    filtered_df = df[df['District'] == district].copy()

    if freq_option == "Monthly":
        filtered_df['Period'] = filtered_df[date_col].dt.to_period('M').astype(str)
    elif freq_option == "Yearly":
        filtered_df['Period'] = filtered_df[date_col].dt.year.astype(str)
    else:
        filtered_df['Period'] = filtered_df[date_col].dt.date.astype(str)

    x_axis_used = 'Period'
    st.markdown(
        f"""
        <h3>
            <img src="https://cdn-icons-png.flaticon.com/512/8451/8451381.png" width="45" style="vertical-align: middle; margin-right: 6px;">
            {district} ({freq_option} view)
        </h3>
        """,
        unsafe_allow_html=True
    )
    
    viz_df = filtered_df.copy()
    default_pairplot = [y_axis]
    if x_axis in numeric_columns and x_axis != y_axis:
        default_pairplot.append(x_axis)

    aggregated_df = viz_df.groupby('Period')[numeric_columns].mean().reset_index()

    tab1, tab2, tab3 = st.tabs(["Overview", "Visualizations", "Advanced Analysis"])
    with tab1:
        #Sample Data
        st.markdown(
            """
            <h3>
                <img src="https://cdn-icons-png.flaticon.com/512/4926/4926731.png " width="40" style="vertical-align: middle; margin-right: 6px;">
                Sample Data
            </h3>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(viz_df.head(), use_container_width=True)

        #Feature Statistics
        st.markdown(
            """
            <h3>
                <img src="https://cdn-icons-png.flaticon.com/512/2318/2318736.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Feature Statistics
            </h3>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(viz_df.select_dtypes(include='number').describe().T, use_container_width=True)

        plot_data = viz_df[[x_axis_used, y_axis]].dropna()
        x_data = plot_data[x_axis_used]
        plot_data[x_axis_used] = pd.to_datetime(x_data, errors='coerce') if pd.api.types.is_datetime64_any_dtype(x_data) else x_data.astype(str)

        
        #Aggregated Data
        st.markdown(
        f"""
        <h3>
            <img src="https://cdn-icons-png.flaticon.com/512/4926/4926731.png" width="40" style="vertical-align: middle; margin-right: 6px;">
            {freq_option} Aggregated Data
            </h3>)
        </h3>
        """,
        unsafe_allow_html=True
        )

        st.dataframe(aggregated_df, use_container_width=True)
        csv = aggregated_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Aggregated CSV",
            data=csv,
            file_name=f"{district.lower().replace(' ', '_')}_{freq_option.lower()}_aggregated.csv",
            mime='text/csv'
        )
        if x_axis in aggregated_df.columns and y_axis in aggregated_df.columns:
                    plot_data = aggregated_df[[x_axis, y_axis]].dropna()
        else:
            st.warning("Selected columns not available in aggregated data.")
            return
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                f"""
                <h4>
                    <img src="https://cdn-icons-png.flaticon.com/512/7837/7837488.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                    Scatter Plot: {y_axis} vs {x_axis}
                </h4>
                """,
                unsafe_allow_html=True
            )

            scatter_chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X(x_axis, title=x_axis),
                y=alt.Y(y_axis, title=y_axis),
                tooltip=[x_axis, y_axis]
            ).interactive().properties(height=300)
            st.altair_chart(scatter_chart, use_container_width=True)

        with col4:
            st.markdown(
                f"""
                <h4>
                    <img src="https://cdn-icons-png.flaticon.com/512/4215/4215828.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                    Box Plot: {y_axis} vs {x_axis}
                </h4>
                """,
                unsafe_allow_html=True
            )
            box_chart = alt.Chart(plot_data).mark_boxplot(extent='min-max').encode(
                y=alt.Y(y_axis, title=y_axis),
                tooltip=[y_axis]
            ).properties(height=300)
            st.altair_chart(box_chart, use_container_width=True)

        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/3586/3586022.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Bar Chart: {y_axis} vs {x_axis}
            </h4>
            """,
            unsafe_allow_html=True
        )
        fig = px.bar(plot_data, x=x_axis, y=y_axis)
        st.plotly_chart(fig, use_container_width=False)


    with tab2:
        # Pull y and Period columns
        plot_data = aggregated_df[[y_axis, 'Period']].dropna().copy()

        # Convert Period to monthly, yearly
        if freq_option == "Monthly":
            plot_data['Period'] = pd.to_datetime(plot_data['Period'], format="%Y-%m", errors="coerce")
        elif freq_option == "Yearly":
            plot_data['Period'] = pd.to_datetime(plot_data['Period'], format="%Y", errors="coerce")
        else:  # Daily
            plot_data['Period'] = pd.to_datetime(plot_data['Period'], errors="coerce")

        # Set appropriate date format 
        x_format = "%Y-%m-%d" if freq_option == "Daily" else "%Y-%m" if freq_option == "Monthly" else "%Y"

        # Choose chart type
        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/7495/7495244.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Line or Area Chart {y_axis} over Time
            </h4>
            """,
            unsafe_allow_html=True
        )
        chart_type = st.radio("Chart Type", ["Line Chart", "Area Chart"], horizontal=True)

        mark = alt.Chart(plot_data).mark_area(opacity=0.5) if chart_type == "Area Chart" else alt.Chart(plot_data).mark_line()

        # Create chart
        chart = mark.encode(
            x=alt.X("Period:T", title="Time", axis=alt.Axis(format=x_format, labelAngle=-45)),
            y=alt.Y(y_axis, title=y_axis),
            tooltip=["Period", y_axis]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        #Distribution
        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/3586/3586022.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                Distribution of Weather Parameters
            </h4>
            """,
            unsafe_allow_html=True
        )
        default_params = [y_axis]
        if x_axis in numeric_columns and x_axis != y_axis:
            default_params.append(x_axis)

        selected_params = st.multiselect("Select Weather Parameters to Visualize", numeric_columns, default=default_params)
        if selected_params:
            for param in selected_params:
                st.markdown(
                    f"""
                    <h4>
                        <img src="https://cdn-icons-png.flaticon.com/512/3586/3586022.png" width="20" style="vertical-align: middle; margin-right: 6px;">
                        Distribution of {param}
                    </h4>
                    """,
                    unsafe_allow_html=True
                )

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
        #Pairplot
        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/3586/3586022.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Pairwise Relationships (Pairplot)
            </h4>
            """,
            unsafe_allow_html=True
        )
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

        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/7837/7837488.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Correlation Between Weather Parameters
            </h4>
            """,
            unsafe_allow_html=True
        )
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
        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/5464/5464694.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Weather Pattern Clustering
            </h4>
            """,
            unsafe_allow_html=True
        )
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

        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/17359/17359250.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Outlier/Anomaly Detection
            </h4>
            """,
            unsafe_allow_html=True
        )
        outlier_col = st.selectbox(
            "Select Variable to Detect Anomalies",
            numeric_columns,
            index=numeric_columns.index(y_axis) if y_axis in numeric_columns else 0,
            key="outlier_var"
        )

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

        st.markdown(
            f"""
            <h4>
                <img src="https://cdn-icons-png.flaticon.com/512/11083/11083363.png" width="40" style="vertical-align: middle; margin-right: 6px;">
                Anomalies Detected
            </h4>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(filtered_df[filtered_df["Anomaly"]][["Date", outlier_col, "Z_Score"]])
