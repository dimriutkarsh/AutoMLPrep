import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score


# Data uploading:

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None


# EDA & Data Cleaning

def perform_eda(df):
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    
    # EDA Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📈 Statistics", "🔍 Patterns", "📉 Outliers", "📋 Data Quality"
    ])
    
    with tab1:
        st.write("#### Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", int(df.shape[0]))
        with col2:
            st.metric("Total Columns", int(df.shape[1]))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        with col4:
            st.metric("Duplicate Rows", int(df.duplicated().sum()))
        
        # Data Types Overview
        st.write("#### Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig = px.pie(values=dtype_counts.values.astype(int), names=dtype_counts.index.astype(str), 
                    title="Data Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.write("#### Statistical Summary")
        
        # Numerical Columns Statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("##### Numerical Columns")
            stats_df = df[numeric_cols].describe().T
            stats_df['variance'] = df[numeric_cols].var()
            stats_df['skewness'] = df[numeric_cols].skew()
            stats_df['kurtosis'] = df[numeric_cols].kurtosis()
            # Convert all values to native Python types
            stats_df = stats_df.applymap(lambda x: float(x) if pd.notnull(x) else x)
            st.dataframe(stats_df.style.format("{:.2f}"))
        
        # Categorical Columns Statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("##### Categorical Columns")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_frequent = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                freq_count = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{col} - Unique Values", int(unique_count))
                with col2:
                    display_text = str(most_frequent)[:20] + "..." if len(str(most_frequent)) > 20 else str(most_frequent)
                    st.metric("Most Frequent", display_text)
                with col3:
                    st.metric("Frequency", int(freq_count))
    
    with tab3:
        st.write("#### Data Patterns & Correlations")
        
        # Correlation Matrix
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            st.write("##### Correlation Heatmap")
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, 
                          aspect="auto",
                          color_continuous_scale='RdBu_r',
                          title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top Correlations
            st.write("##### Top Correlations")
            corr_pairs = corr_matrix.unstack().sort_values(key=lambda x: abs(x), ascending=False)
            top_corrs = corr_pairs[corr_pairs != 1.0].head(10)
            for (col1, col2), value in top_corrs.items():
                st.write(f"**{col1}** & **{col2}**: {value:.3f}")
    
    with tab4:
        st.write("#### Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            if selected_col:
                Q1 = float(df[selected_col].quantile(0.25))
                Q3 = float(df[selected_col].quantile(0.75))
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Lower Bound", f"{lower_bound:.2f}")
                with col2:
                    st.metric("Upper Bound", f"{upper_bound:.2f}")
                with col3:
                    st.metric("Outlier Count", int(len(outliers)))
                with col4:
                    outlier_pct = (len(outliers)/len(df))*100
                    st.metric("Outlier %", f"{outlier_pct:.2f}%")
                
                # Outlier Visualization
                fig = px.box(df, y=selected_col, title=f"Outlier Analysis - {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.write("#### Data Quality Report")
        
        quality_data = []
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            missing_pct = (missing_count / len(df)) * 100
            unique_count = int(df[col].nunique())
            dtype = str(df[col].dtype)
            
            quality_data.append({
                'Column': col,
                'Data Type': dtype,
                'Missing Values': missing_count,
                'Missing %': f"{missing_pct:.2f}%",
                'Unique Values': unique_count,
                'Completeness': f"{(100 - missing_pct):.2f}%"
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df.style.background_gradient(subset=['Missing Values'], cmap='Reds'))

def clean_data(df):
    st.subheader("🧹 Advanced Data Cleaning")
    
    # Create tabs for different cleaning operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Overview", "❌ Missing Values", "🔁 Data Types", "📊 Outliers", "✅ Final Report"
    ])
    
    df_clean = df.copy()
    cleaning_log = []
    
    with tab1:
        st.write("### Dataset Overview Before Cleaning")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Rows", int(df.shape[0]))
        with col2:
            st.metric("Original Columns", int(df.shape[1]))
        with col3:
            st.metric("Total Missing", int(df.isnull().sum().sum()))
        with col4:
            st.metric("Duplicate Rows", int(df.duplicated().sum()))
        
        # Missing values heatmap
        st.write("#### Missing Values Visualization")
        if df.isnull().sum().sum() > 0:
            fig = px.imshow(df.isnull(), 
                          aspect="auto",
                          color_continuous_scale=['green', 'red'],
                          title="Missing Values Heatmap (Red = Missing)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")
    
    with tab2:
        st.write("### Missing Values Treatment")
        
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        
        if missing_cols:
            st.write("#### Columns with Missing Values")
            for col in missing_cols:
                missing_count = int(df_clean[col].isnull().sum())
                missing_pct = (missing_count / len(df_clean)) * 100
                
                st.write(f"**{col}** - {missing_count} missing values ({missing_pct:.2f}%)")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    treatment_method = st.selectbox(
                        f"Treatment for {col}",
                        ["Do Nothing", "Drop Column", "Fill with Mean", "Fill with Median", 
                         "Fill with Mode", "Fill with Specific Value", "Forward Fill", "Backward Fill"],
                        key=f"treatment_{col}"
                    )
                
                with col2:
                    if treatment_method == "Fill with Specific Value":
                        fill_value = st.text_input("Fill value", key=f"fill_{col}")
                    elif treatment_method in ["Forward Fill", "Backward Fill"]:
                        st.info("Will fill missing values using adjacent rows")
                
                with col3:
                    if st.button("Apply", key=f"apply_{col}"):
                        if treatment_method == "Drop Column":
                            df_clean.drop(columns=[col], inplace=True)
                            cleaning_log.append(f"🗑️ Dropped column '{col}' ({missing_count} missing values)")
                        
                        elif treatment_method == "Fill with Mean":
                            if df_clean[col].dtype in ['float64', 'int64']:
                                fill_val = float(df_clean[col].mean())
                                df_clean[col].fillna(fill_val, inplace=True)
                                cleaning_log.append(f"📊 Filled '{col}' with mean: {fill_val:.2f}")
                        
                        elif treatment_method == "Fill with Median":
                            if df_clean[col].dtype in ['float64', 'int64']:
                                fill_val = float(df_clean[col].median())
                                df_clean[col].fillna(fill_val, inplace=True)
                                cleaning_log.append(f"📊 Filled '{col}' with median: {fill_val:.2f}")
                        
                        elif treatment_method == "Fill with Mode":
                            if not df_clean[col].mode().empty:
                                fill_val = df_clean[col].mode()[0]
                                df_clean[col].fillna(fill_val, inplace=True)
                                cleaning_log.append(f"📊 Filled '{col}' with mode: {fill_val}")
                        
                        elif treatment_method == "Fill with Specific Value" and fill_value:
                            # Try to convert to numeric, otherwise keep as string
                            try:
                                fill_val = float(fill_value)
                            except ValueError:
                                fill_val = fill_value
                            df_clean[col].fillna(fill_val, inplace=True)
                            cleaning_log.append(f"📝 Filled '{col}' with: {fill_value}")
                        
                        elif treatment_method == "Forward Fill":
                            df_clean[col].fillna(method='ffill', inplace=True)
                            cleaning_log.append(f"⬇️ Forward filled '{col}'")
                        
                        elif treatment_method == "Backward Fill":
                            df_clean[col].fillna(method='bfill', inplace=True)
                            cleaning_log.append(f"⬆️ Backward filled '{col}'")
                        
                        st.success(f"Applied {treatment_method} to {col}")
        else:
            st.success("✅ No missing values to treat!")
    
    with tab3:
        st.write("### Data Type Conversion")
        
        st.write("#### Current Data Types")
        dtype_df = pd.DataFrame({
            'Column': df_clean.columns,
            'Current Type': df_clean.dtypes.astype(str),
        })
        st.dataframe(dtype_df)
        
        for idx, col in enumerate(df_clean.columns):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"**{col}**")
            with col2:
                new_dtype = st.selectbox(
                    "Convert to",
                    ["Keep Original", "numeric", "string", "category", "datetime"],
                    key=f"dtype_{col}"
                )
            with col3:
                if st.button("Convert", key=f"convert_{col}"):
                    try:
                        if new_dtype == "numeric":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                            cleaning_log.append(f"🔢 Converted '{col}' to numeric")
                        elif new_dtype == "string":
                            df_clean[col] = df_clean[col].astype(str)
                            cleaning_log.append(f"📝 Converted '{col}' to string")
                        elif new_dtype == "category":
                            df_clean[col] = df_clean[col].astype('category')
                            cleaning_log.append(f"🏷️ Converted '{col}' to category")
                        elif new_dtype == "datetime":
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                            cleaning_log.append(f"📅 Converted '{col}' to datetime")
                        st.success(f"Converted {col} to {new_dtype}")
                    except Exception as e:
                        st.error(f"Error converting {col}: {str(e)}")
    
    with tab4:
        st.write("### Outlier Treatment")
        
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for outlier treatment", numeric_cols)
            
            if selected_col:
                # Calculate outliers
                Q1 = float(df_clean[selected_col].quantile(0.25))
                Q3 = float(df_clean[selected_col].quantile(0.75))
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = int(((df_clean[selected_col] < lower_bound) | 
                               (df_clean[selected_col] > upper_bound)).sum())
                
                st.write(f"**Outliers detected:** {outlier_count}")
                
                treatment = st.selectbox(
                    "Outlier treatment method",
                    ["Do Nothing", "Remove Outliers", "Cap Outliers", "Transform (log)"]
                )
                
                if st.button("Apply Outlier Treatment"):
                    if treatment == "Remove Outliers":
                        original_count = len(df_clean)
                        df_clean = df_clean[
                            (df_clean[selected_col] >= lower_bound) & 
                            (df_clean[selected_col] <= upper_bound)
                        ]
                        removed_count = original_count - len(df_clean)
                        cleaning_log.append(f"🗑️ Removed {removed_count} outliers from '{selected_col}'")
                        st.success(f"Removed {removed_count} outliers")
                    
                    elif treatment == "Cap Outliers":
                        df_clean[selected_col] = np.where(
                            df_clean[selected_col] > upper_bound, upper_bound,
                            np.where(df_clean[selected_col] < lower_bound, lower_bound, 
                                   df_clean[selected_col])
                        )
                        cleaning_log.append(f"📏 Capped outliers in '{selected_col}'")
                        st.success("Outliers capped successfully")
                    
                    elif treatment == "Transform (log)":
                        if (df_clean[selected_col] > 0).all():
                            df_clean[selected_col] = np.log1p(df_clean[selected_col])
                            cleaning_log.append(f"📐 Applied log transform to '{selected_col}'")
                            st.success("Log transform applied successfully")
                        else:
                            st.error("Log transform requires all positive values")
        else:
            st.info("No numeric columns available for outlier treatment")
    
    with tab5:
        st.write("### Cleaning Summary Report")
        
        # Display cleaning log
        if cleaning_log:
            st.write("#### Operations Performed")
            for log_entry in cleaning_log:
                st.write(f"• {log_entry}")
        else:
            st.info("No cleaning operations performed yet")
        
        # Before-After Comparison
        col1, col2 = st.columns(2)
        with col1:
            st.write("##### Before Cleaning")
            st.metric("Rows", int(df.shape[0]))
            st.metric("Columns", int(df.shape[1]))
            st.metric("Missing Values", int(df.isnull().sum().sum()))
            st.metric("Duplicates", int(df.duplicated().sum()))
        
        with col2:
            st.write("##### After Cleaning")
            # Convert all numpy types to native Python types for delta calculations
            rows_before = int(df.shape[0])
            rows_after = int(df_clean.shape[0])
            cols_before = int(df.shape[1])
            cols_after = int(df_clean.shape[1])
            missing_before = int(df.isnull().sum().sum())
            missing_after = int(df_clean.isnull().sum().sum())
            dupes_before = int(df.duplicated().sum())
            dupes_after = int(df_clean.duplicated().sum())
            
            st.metric("Rows", rows_after, delta=rows_after - rows_before)
            st.metric("Columns", cols_after, delta=cols_after - cols_before)
            st.metric("Missing Values", missing_after, delta=missing_after - missing_before)
            st.metric("Duplicates", dupes_after, delta=dupes_after - dupes_before)
        
        # Data Quality Score
        quality_score = calculate_quality_score(df_clean)
        st.write(f"#### Data Quality Score: {quality_score:.1f}/100")
        st.progress(float(quality_score/100))
        
        # Download cleaned data
        csv = df_clean.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
    
    return df_clean

def calculate_quality_score(df):
    """Calculate a comprehensive data quality score"""
    total_score = 100
    
    # Penalty for missing values
    missing_penalty = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 50
    total_score -= missing_penalty
    
    # Penalty for duplicates
    duplicate_penalty = (df.duplicated().sum() / df.shape[0]) * 20
    total_score -= duplicate_penalty
    
    # Bonus for data type consistency
    consistent_dtypes = len(set(df.dtypes)) / len(df.dtypes) * 10
    total_score += consistent_dtypes
    
    return max(0, min(100, float(total_score)))

# Add this to your main app
def show_data_cleaning_help():
    with st.expander("📖 Data Cleaning Guide"):
        st.markdown("""
        **Data Cleaning Best Practices:**
        
        - **Missing Values**: 
          - Use mean/median for numerical data
          - Use mode for categorical data
          - Consider dropping if >30% missing
        
        - **Outliers**:
          - Remove if clearly erroneous
          - Cap if valid but extreme
          - Transform to reduce skewness
        
        - **Data Types**:
          - Convert to appropriate types for analysis
          - Use categories for low-cardinality text data
          - Ensure datetime formats are consistent
        """)


# Data  Visualization:

def visualize_data(df):
    st.subheader("📊 Advanced Data Visualization Explorer")
    
    # Sidebar for advanced settings
    st.sidebar.subheader("🎛️ Visualization Settings")
    
    # Theme selection
    theme = st.sidebar.selectbox(
        "Color Theme",
        ["default", "dark", "white", "pastel", "bright"]
    )
    
    # Figure size control
    fig_width = st.sidebar.slider("Figure Width", 6, 16, 10)
    fig_height = st.sidebar.slider("Figure Height", 4, 12, 6)
    
    # Set theme
    if theme == "dark":
        plt.style.use('dark_background')
        plotly_template = "plotly_dark"
    elif theme == "white":
        plt.style.use('default')
        plotly_template = "plotly_white"
    elif theme == "pastel":
        plotly_template = "seaborn"
    elif theme == "bright":
        plotly_template = "plotly"
    else:
        plotly_template = "plotly"

    # Enhanced graph selection
    graph_type = st.selectbox(
        "Select Graph Type",
        [
            "Histogram", "Box Plot", "Violin Plot", "Scatter Plot", 
            "Line Plot", "Bar Plot", "Horizontal Bar Plot", "Area Plot",
            "Heatmap (correlation)", "Pairplot", "Pie Chart", "Count Plot",
            "Density Plot", "3D Scatter Plot", "Bubble Chart", "Treemap",
            "Swarm Plot"
        ]
    )

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Common customization options
    st.sidebar.subheader("🎨 Customization")
    show_grid = st.sidebar.checkbox("Show Grid", True)
    transparency = st.sidebar.slider("Transparency", 0.1, 1.0, 0.8)

    if graph_type == "Histogram":
        col = st.selectbox("Select column", numeric_cols)
        bins = st.slider("Number of bins", 5, 100, 30)
        show_kde = st.checkbox("Show KDE", True)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.histplot(df[col], kde=show_kde, bins=bins, alpha=transparency, ax=ax)
        ax.grid(show_grid)
        ax.set_title(f'Histogram of {col}')
        st.pyplot(fig)

    elif graph_type == "Box Plot":
        col = st.selectbox("Select column", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if hue_col:
            sns.boxplot(data=df, x=hue_col, y=col, ax=ax)
        else:
            sns.boxplot(x=df[col], ax=ax)
        ax.grid(show_grid)
        st.pyplot(fig)

    elif graph_type == "Violin Plot":
        col = st.selectbox("Select numeric column", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if hue_col:
            sns.violinplot(data=df, x=hue_col, y=col, ax=ax)
        else:
            sns.violinplot(y=df[col], ax=ax)
        ax.grid(show_grid)
        st.pyplot(fig)

    elif graph_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X Axis", numeric_cols)
            size_col = st.selectbox("Size variable (optional)", [None] + numeric_cols)
        with col2:
            y_axis = st.selectbox("Y Axis", numeric_cols)
            color_col = st.selectbox("Color variable (optional)", [None] + all_cols)
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            size=size_col, color=color_col,
            opacity=transparency,
            template=plotly_template,
            title=f"Scatter Plot: {x_axis} vs {y_axis}"
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig)

    elif graph_type == "Line Plot":
        x_axis = st.selectbox("X Axis", all_cols)
        y_axis = st.selectbox("Y Axis", numeric_cols)
        color_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
        
        fig = px.line(
            df, x=x_axis, y=y_axis, color=color_col,
            template=plotly_template,
            title=f"Line Plot: {y_axis} over {x_axis}"
        )
        st.plotly_chart(fig)

    elif graph_type == "Bar Plot":
        x_axis = st.selectbox("X Axis", all_cols)
        y_axis = st.selectbox("Y Axis", numeric_cols)
        color_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
        
        fig = px.bar(
            df, x=x_axis, y=y_axis, color=color_col,
            template=plotly_template,
            title=f"Bar Plot: {y_axis} by {x_axis}"
        )
        st.plotly_chart(fig)

    elif graph_type == "Horizontal Bar Plot":
        x_axis = st.selectbox("Value Axis", numeric_cols)
        y_axis = st.selectbox("Category Axis", all_cols)
        
        # Take top N categories to avoid overcrowding
        top_n = st.slider("Show top N categories", 5, 50, 15)
        top_categories = df[y_axis].value_counts().head(top_n).index
        filtered_df = df[df[y_axis].isin(top_categories)]
        
        fig = px.bar(
            filtered_df, y=y_axis, x=x_axis, 
            orientation='h',
            template=plotly_template,
            title=f"Horizontal Bar Plot: Top {top_n} {y_axis}"
        )
        st.plotly_chart(fig)

    elif graph_type == "Area Plot":
        x_axis = st.selectbox("X Axis", all_cols)
        y_axis = st.selectbox("Y Axis", numeric_cols)
        
        fig = px.area(
            df, x=x_axis, y=y_axis,
            template=plotly_template,
            title=f"Area Plot: {y_axis} over {x_axis}"
        )
        st.plotly_chart(fig)

    elif graph_type == "Heatmap (correlation)":
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns for correlation.")

    elif graph_type == "Pairplot":
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.info("Rendering pairplot... might take time for large datasets.")
            hue_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
            fig = sns.pairplot(df, hue=hue_col, vars=numeric_cols[:5], diag_kind='kde')  # Limit to 5 cols for performance
            st.pyplot(fig)
        else:
            st.warning("No numeric columns for pairplot.")

    elif graph_type == "Pie Chart":
        col = st.selectbox("Select categorical column", categorical_cols)
        top_n = st.slider("Show top N categories", 5, 20, 10)
        
        value_counts = df[col].value_counts().head(top_n)
        fig = px.pie(
            values=value_counts.values.astype(int),
            names=value_counts.index.astype(str),
            title=f"Pie Chart: {col} (Top {top_n})"
        )
        st.plotly_chart(fig)

    elif graph_type == "Count Plot":
        col = st.selectbox("Select categorical column", categorical_cols)
        top_n = st.slider("Show top N categories", 5, 50, 15)
        
        top_categories = df[col].value_counts().head(top_n).index
        filtered_df = df[df[col].isin(top_categories)]
        
        fig = px.histogram(
            filtered_df, x=col, 
            title=f"Count Plot: {col} (Top {top_n})",
            template=plotly_template
        )
        st.plotly_chart(fig)

    elif graph_type == "Density Plot":
        col = st.selectbox("Select column", numeric_cols)
        hue_col = st.selectbox("Group by (optional)", [None] + categorical_cols)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if hue_col:
            for category in df[hue_col].unique():
                subset = df[df[hue_col] == category]
                sns.kdeplot(subset[col], label=str(category), ax=ax)
            ax.legend()
        else:
            sns.kdeplot(df[col], ax=ax)
        ax.grid(show_grid)
        ax.set_title(f'Density Plot of {col}')
        st.pyplot(fig)

    elif graph_type == "3D Scatter Plot":
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X Axis", numeric_cols, key='3d_x')
        with col2:
            y_axis = st.selectbox("Y Axis", numeric_cols, key='3d_y')
        with col3:
            z_axis = st.selectbox("Z Axis", numeric_cols, key='3d_z')
        
        color_col = st.selectbox("Color by", [None] + all_cols, key='3d_color')
        
        fig = px.scatter_3d(
            df, x=x_axis, y=y_axis, z=z_axis, color=color_col,
            template=plotly_template,
            title=f"3D Scatter Plot"
        )
        st.plotly_chart(fig)

    elif graph_type == "Bubble Chart":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X Axis", numeric_cols, key='bubble_x')
            size_col = st.selectbox("Bubble Size", numeric_cols, key='bubble_size')
        with col2:
            y_axis = st.selectbox("Y Axis", numeric_cols, key='bubble_y')
            color_col = st.selectbox("Bubble Color", [None] + all_cols, key='bubble_color')
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis, size=size_col, color=color_col,
            hover_name=df.index if df.index.name else None,
            size_max=60,
            template=plotly_template,
            title="Bubble Chart"
        )
        st.plotly_chart(fig)

    elif graph_type == "Treemap":
        if len(categorical_cols) >= 2:
            path_cols = st.multiselect("Select hierarchy columns", categorical_cols, max_selections=3)
            value_col = st.selectbox("Value column", [None] + numeric_cols)
            
            if path_cols:
                if value_col:
                    fig = px.treemap(df, path=path_cols, values=value_col)
                else:
                    fig = px.treemap(df, path=path_cols)
                st.plotly_chart(fig)
        else:
            st.warning("Need at least 2 categorical columns for treemap.")

    elif graph_type == "Swarm Plot":
        if numeric_cols and categorical_cols:
            numeric_col = st.selectbox("Numeric column", numeric_cols)
            category_col = st.selectbox("Category column", categorical_cols)
            
            # Limit categories for performance
            top_categories = df[category_col].value_counts().head(10).index
            filtered_df = df[df[category_col].isin(top_categories)]
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.swarmplot(data=filtered_df, x=category_col, y=numeric_col, ax=ax)
            ax.grid(show_grid)
            st.pyplot(fig)
        else:
            st.warning("Need both numeric and categorical columns for swarm plot.")

    # Download option for the plot
    st.sidebar.subheader("💾 Export")
    if st.sidebar.button("Save Current Plot"):
        st.sidebar.info("Plot saving feature can be implemented with plt.savefig() or plotly's write_image()")

# Additional helper function for data summary
def show_visualization_help():
    with st.expander("📖 Visualization Guide"):
        st.markdown("""
        **Chart Selection Guide:**
        - **Histogram**: Distribution of numerical data
        - **Box Plot**: Statistical summary and outliers
        - **Violin Plot**: Distribution + density
        - **Scatter Plot**: Relationship between two variables
        - **3D Scatter**: Three-dimensional relationships
        - **Bubble Chart**: Scatter plot with size dimension
        - **Heatmap**: Correlation matrix
        - **Pairplot**: All pairwise relationships
        - **Treemap**: Hierarchical data representation
        """)


 #Model Training code here:

def ml_training(df):
    st.subheader("🤖 Enhanced ML Model Training")

    columns = df.columns.tolist()
    target = st.selectbox("Select target variable", columns)

    if target:
        # Store original data for later use
        original_df = df.copy()
        
        # Model selection
        model_type = st.selectbox(
            "Select Model Type",
            ["Auto Detect", "Classification", "Regression"]
        )
        
        if model_type == "Auto Detect":
            if df[target].dtype == 'object' or df[target].nunique() < 10:
                is_classification = True
                st.info("🔍 Auto-detected: Classification problem")
            else:
                is_classification = False
                st.info("🔍 Auto-detected: Regression problem")
        elif model_type == "Classification":
            is_classification = True
        else:
            is_classification = False

        # Data preprocessing section
        st.write("### 🔧 Data Preprocessing")
        
        df_ml = df.copy()
        target_encoder = None
        
        # Check target variable encoding
        if is_classification:
            st.write("#### Target Variable Analysis (Classification)")
            
            # Check if target needs encoding
            if df_ml[target].dtype == 'object' or df_ml[target].nunique() > 10:
                st.warning("⚠️ Target variable needs encoding for classification")
                
                # Create encoded version
                target_encoded = f"{target}_encoded"
                target_encoder = LabelEncoder()
                df_ml[target_encoded] = target_encoder.fit_transform(df_ml[target].astype(str))
                
                st.info(f"✅ Created encoded target column: '{target_encoded}'")
                st.write("**Encoding mapping:**")
                classes_df = pd.DataFrame({
                    'Original Class': target_encoder.classes_,
                    'Encoded Value': range(len(target_encoder.classes_))
                })
                st.dataframe(classes_df)
                
                # Use encoded target for modeling
                modeling_target = target_encoded
            else:
                modeling_target = target
                # Create encoder for original target values
                target_encoder = LabelEncoder()
                target_encoder.fit(df_ml[target])
                st.success("✅ Target variable is already properly encoded")
                
        else:  # Regression
            st.write("#### Target Variable Analysis (Regression)")
            
            # Check if target needs scaling for regression
            if df_ml[target].dtype == 'object':
                st.error("❌ Cannot use categorical target for regression. Please select a numerical target.")
                return None, None, None, None, None, None, None
            
            # Check if scaling would be beneficial
            target_std = df_ml[target].std()
            if target_std > 100:  # Arbitrary threshold for large scales
                st.warning("⚠️ Target variable has large scale. Consider scaling for better model performance.")
                scale_target = st.checkbox("Apply Standard Scaling to target variable", value=False)
                
                if scale_target:
                    scaler_target = StandardScaler()
                    target_scaled = f"{target}_scaled"
                    df_ml[target_scaled] = scaler_target.fit_transform(df_ml[[target]])
                    modeling_target = target_scaled
                    st.success(f"✅ Applied Standard Scaling to target. Using '{target_scaled}'")
                else:
                    modeling_target = target
            else:
                modeling_target = target
                st.success("✅ Target variable scale is appropriate for regression")

        # Handle feature columns preprocessing
        st.write("#### Feature Variables Preprocessing")
        
        feature_cols = [col for col in df_ml.columns if col not in [target, modeling_target]]
        categorical_features = []
        numerical_features = []
        
        for col in feature_cols:
            if df_ml[col].dtype == 'object' or (df_ml[col].nunique() < 10 and df_ml[col].dtype != 'float64'):
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        st.write(f"**Categorical features ({len(categorical_features)}):** {categorical_features}")
        st.write(f"**Numerical features ({len(numerical_features)}):** {numerical_features}")

        # Preprocessing options
        preprocessing_method = st.selectbox(
            "Select preprocessing method",
            ["Auto Preprocessing", "Manual Preprocessing"]
        )
        
        encoders = {}
        scalers = {}
        
        if preprocessing_method == "Auto Preprocessing":
            # Auto preprocessing for categorical features
            if categorical_features:
                st.info("🔄 Applying Label Encoding to categorical features")
                for col in categorical_features:
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                    encoders[col] = le
                
            # Auto preprocessing for numerical features
            if numerical_features:
                # Check if scaling is needed
                needs_scaling = any(df_ml[col].std() > 100 for col in numerical_features)
                if needs_scaling:
                    st.info("🔄 Applying Standard Scaling to numerical features")
                    for col in numerical_features:
                        scaler = StandardScaler()
                        df_ml[col] = scaler.fit_transform(df_ml[[col]])
                        scalers[col] = scaler
        else:
            # Manual preprocessing options
            st.write("##### Manual Preprocessing Options")
            
            # Categorical features encoding
            if categorical_features:
                cat_method = st.selectbox(
                    "Categorical encoding method",
                    ["Label Encoding", "One-Hot Encoding"]
                )
                
                if cat_method == "Label Encoding":
                    for col in categorical_features:
                        le = LabelEncoder()
                        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                        encoders[col] = le
                else:  # One-Hot Encoding
                    df_ml = pd.get_dummies(df_ml, columns=categorical_features, drop_first=True)
            
            # Numerical features scaling
            if numerical_features:
                scale_numerical = st.checkbox("Apply scaling to numerical features", value=True)
                if scale_numerical:
                    scale_method = st.selectbox(
                        "Scaling method",
                        ["Standard Scaler", "MinMax Scaler"]
                    )
                    
                    for col in numerical_features:
                        if scale_method == "Standard Scaler":
                            scaler = StandardScaler()
                        else:
                            scaler = MinMaxScaler()
                        
                        df_ml[col] = scaler.fit_transform(df_ml[[col]])
                        scalers[col] = scaler

        # Prepare features and target
        X = df_ml.drop([target, modeling_target] if modeling_target != target else [target], axis=1)
        y = df_ml[modeling_target]

        # Train-test split
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Auto Model Selection
        st.write("### 🤖 Auto Model Selection")
        
        if is_classification:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
            }

        # Train and evaluate all models
        results = {}
        best_model_name = None
        best_score = -float('inf') if is_classification else float('inf')
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if is_classification:
                    score = accuracy_score(y_test, y_pred)
                    results[name] = {
                        'model': model,
                        'accuracy': score,
                        'predictions': y_pred
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_model_name = name
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    results[name] = {
                        'model': model,
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2': r2_score(y_test, y_pred),
                        'predictions': y_pred
                    }
                    
                    if mse < best_score:  # Lower MSE is better
                        best_score = mse
                        best_model_name = name
                        
            except Exception as e:
                st.warning(f"Could not train {name}: {str(e)}")
                continue
            
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text("Training completed!")
        
        # Display model comparison
        st.write("#### Model Comparison")
        if is_classification:
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[name]['accuracy'] for name in results.keys()]
            }).sort_values('Accuracy', ascending=False)
        else:
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'MSE': [results[name]['mse'] for name in results.keys()],
                'RMSE': [results[name]['rmse'] for name in results.keys()],
                'R²': [results[name]['r2'] for name in results.keys()]
            }).sort_values('MSE', ascending=True)
        
        st.dataframe(comparison_df.style.format({
            'Accuracy': '{:.4f}',
            'MSE': '{:.4f}',
            'RMSE': '{:.4f}',
            'R²': '{:.4f}'
        }))
        
        st.success(f"🎯 Best Model: **{best_model_name}**")
        
        # Use the best model for detailed evaluation
        best_result = results[best_model_name]
        model = best_result['model']
        y_pred = best_result['predictions']

        # Detailed evaluation of best model
        st.write(f"### 📊 Detailed Evaluation - {best_model_name}")
        
        if is_classification:
            accuracy = best_result['accuracy']
            st.success(f"**Accuracy:** {accuracy:.4f}")
            
            # Confusion Matrix
            st.write("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Classification Report
            st.write("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"))
        else:
            mse = best_result['mse']
            rmse = best_result['rmse']
            r2 = best_result['r2']
            
            st.success(f"**MSE:** {mse:.4f}")
            st.success(f"**RMSE:** {rmse:.4f}")
            st.success(f"**R² Score:** {r2:.4f}")
            
            # Actual vs Predicted plot
            st.write("#### Actual vs Predicted")
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
            fig.add_shape(type='line', line=dict(dash='dash'), 
                         x0=float(y_test.min()), y0=float(y_test.min()), 
                         x1=float(y_test.max()), y1=float(y_test.max()))
            st.plotly_chart(fig)

        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            st.write("#### Feature Importance")
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='importance', y='feature', 
                        orientation='h', title="Top 10 Feature Importance")
            st.plotly_chart(fig)

        return model, X.columns.tolist(), is_classification, encoders, target_encoder, original_df, target

    return None, None, None, None, None, None, None

# Model Testing Function
def test_model(model, features, is_classification, target_encoder, original_df, target_col):
    st.subheader("🧪 Model Testing")
    
    # Store original feature mappings for categorical columns
    feature_mappings = {}
    for feature in features:
        if feature in original_df.columns and original_df[feature].dtype == 'object':
            feature_mappings[feature] = {
                'options': original_df[feature].unique().tolist(),
                'encoder': LabelEncoder().fit(original_df[feature])
            }
    
    st.write("### Input Features for Prediction")
    
    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            if feature in feature_mappings:
                # Categorical feature - show dropdown
                selected_option = st.selectbox(
                    f"{feature}",
                    options=feature_mappings[feature]['options'],
                    key=f"test_{feature}"
                )
                # Convert to encoded value
                input_data[feature] = feature_mappings[feature]['encoder'].transform([selected_option])[0]
            else:
                # Numerical feature - show number input
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=float(original_df[feature].mean()) if len(original_df) > 0 else 0.0,
                    step=0.1,
                    key=f"test_{feature}"
                )
    
    if st.button("🔮 Predict"):
        try:
            # Create input array
            input_array = np.array([[input_data[feature] for feature in features]])
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            
            if is_classification:
                # Create meaningful labels for binary classification
                if target_encoder is not None:
                    original_prediction = target_encoder.inverse_transform([int(prediction)])[0]
                else:
                    # For binary classification like HeartDisease (0,1)
                    if target_col in original_df.columns:
                        unique_values = original_df[target_col].unique()
                        if len(unique_values) == 2:
                            # Map 0/1 to meaningful labels
                            if set(unique_values) == {0, 1}:
                                label_map = {0: "No Heart Disease", 1: "Heart Disease"}
                            else:
                                label_map = {i: str(val) for i, val in enumerate(sorted(unique_values))}
                            original_prediction = label_map.get(int(prediction), str(prediction))
                        else:
                            original_prediction = str(prediction)
                    else:
                        original_prediction = str(prediction)
                
                st.success(f"🎯 Prediction: **{original_prediction}**")
                
            else:
                st.success(f"🎯 Prediction: **{float(prediction):.2f}**")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

#main app is form here:
def main():
    st.set_page_config(
        page_title="AutoML Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 AutoML + EDA Dashboard")
    st.markdown("### Upload your data, explore, clean, visualize, and build ML models - all in one place!")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            # Sidebar Navigation
            st.sidebar.title("📌 Navigation")
            section = st.sidebar.radio("Choose Section", 
                ["Data Overview", "EDA", "Data Cleaning", "Visualization", "Model Training", "Model Testing"]
            )

            if section == "Data Overview":
                st.write("### 📋 Data Overview")
                st.dataframe(df.head())
                st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Basic info
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### Data Types")
                    dtype_summary = df.dtypes.value_counts()
                    for dtype, count in dtype_summary.items():
                        st.write(f"- {dtype}: {int(count)} columns")
                
                with col2:
                    st.write("#### Missing Values Summary")
                    missing_total = df.isnull().sum().sum()
                    st.write(f"Total missing values: {int(missing_total)}")
                    if missing_total > 0:
                        st.write("Columns with missing values:")
                        missing_cols = df.columns[df.isnull().any()].tolist()
                        for col in missing_cols:
                            missing_count = df[col].isnull().sum()
                            st.write(f"- {col}: {int(missing_count)}")

            elif section == "EDA":
                perform_eda(df)

            elif section == "Data Cleaning":
                show_data_cleaning_help()
                df_clean = clean_data(df)
                st.session_state["df_clean"] = df_clean

            elif section == "Visualization":
                show_visualization_help()
                if "df_clean" in st.session_state:
                    visualize_data(st.session_state["df_clean"])
                else:
                    st.warning("⚠️ Please clean your data first in the 'Data Cleaning' section.")
                    if st.button("Use Original Data for Visualization"):
                        visualize_data(df)

            elif section == "Model Training":
                if "df_clean" in st.session_state:
                    model, features, is_classification, encoders, target_encoder, original_df, target_col = ml_training(st.session_state["df_clean"])
                    if model is not None:
                        st.session_state["model"] = model
                        st.session_state["features"] = features
                        st.session_state["is_classification"] = is_classification
                        st.session_state["encoders"] = encoders
                        st.session_state["target_encoder"] = target_encoder
                        st.session_state["original_df"] = original_df
                        st.session_state["target_col"] = target_col
                else:
                    st.warning("⚠️ Please clean your data first in the 'Data Cleaning' section.")
                    if st.button("Use Original Data for Model Training"):
                        model, features, is_classification, encoders, target_encoder, original_df, target_col = ml_training(df)
                        if model is not None:
                            st.session_state["model"] = model
                            st.session_state["features"] = features
                            st.session_state["is_classification"] = is_classification
                            st.session_state["encoders"] = encoders
                            st.session_state["target_encoder"] = target_encoder
                            st.session_state["original_df"] = original_df
                            st.session_state["target_col"] = target_col

            elif section == "Model Testing":
                if "model" in st.session_state and "features" in st.session_state:
                    test_model(
                        st.session_state["model"], 
                        st.session_state["features"],
                        st.session_state.get("is_classification", False),
                        st.session_state.get("target_encoder", None),
                        st.session_state.get("original_df", None),
                        st.session_state.get("target_col", None)
                    )
                else:
                    st.warning("⚠️ Please train a model first in the 'Model Training' section.")

    else:
        st.info("👆 Please upload a CSV or Excel file to get started!")
        
        # Sample data option
        if st.button("🎲 Load Sample Dataset"):
            # Create sample data
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'Age': np.random.randint(18, 65, 100),
                'Income': np.random.normal(50000, 15000, 100),
                'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
                'Experience': np.random.randint(0, 40, 100),
                'Salary': np.random.normal(70000, 20000, 100),
                'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 100)
            })
            st.session_state["sample_data"] = sample_data
            st.success("✅ Sample dataset loaded! You can now explore all features.")
            st.dataframe(sample_data.head())

if __name__ == "__main__":
    main()