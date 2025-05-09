# visualization_handler.py

import pandas as pd
import plotly.express as px
import streamlit as st


def plot_distribution(df: pd.DataFrame, value_col: str, value_label: str):
    """Plots a histogram of a specified value column."""
    if df is None or df.empty or value_col not in df.columns or df[value_col].isnull().all():
        st.warning(f"No valid data in '{value_col}' column for distribution plot.")
        return

    print(f"Generating distribution histogram for {value_label}.")
    fig = px.histogram(df.dropna(subset=[value_col]), x=value_col, nbins=50, title=f'Distribution of {value_label}')
    st.plotly_chart(fig, use_container_width=True)

def plot_value_vs_feature(df: pd.DataFrame, feature_col: str, value_col: str, color_col: str, feature_label: str, value_label: str, color_label: str):
    """Plots a specified value column vs. a feature column (scatter plot), colored by a category."""
    required_cols = [value_col, feature_col, color_col]
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        st.warning(f"Data or required columns ({', '.join(required_cols)}) not found for scatter plot.")
        return
    # Drop rows with NaNs in key plotting columns
    df_filtered = df.dropna(subset=[value_col, feature_col, color_col])
    if df_filtered.empty:
         st.warning(f"No valid data points after dropping NaNs for '{value_label}' vs '{feature_label}' plot.")
         return


    print(f"Generating {value_label} vs {feature_label} scatter plot colored by {color_label}.")
    fig = px.scatter(
        df_filtered,
        x=feature_col,
        y=value_col,
        color=color_col,
        labels={value_col: value_label, feature_col: feature_label, color_col: color_label}, # Set labels
        hover_data=['customer_id', value_col, feature_col, color_col] if 'customer_id' in df_filtered.columns else [value_col, feature_col, color_col], # Add hover data if customer_id exists
        title=f'{value_label} vs. {feature_label} colored by {color_label}'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_average_value_by_category(df: pd.DataFrame, category_col: str, value_col: str, category_label: str, value_label: str):
    """Plots average of a specified value column by a categorical column (bar chart)."""
    required_cols = [value_col, category_col]
    if df is None or df.empty or not all(col in df.columns for col in required_cols):
        st.warning(f"Data or required columns ({', '.join(required_cols)}) not found for average bar chart.")
        return
    if not pd.api.types.is_categorical_dtype(df[category_col]) and not df[category_col].dtype == 'object':
         st.warning(f"Category column '{category_col}' is not categorical or object type.")
         return
    # Drop rows where the category or value is missing for the group by
    df_filtered = df.dropna(subset=[value_col, category_col])
    if df_filtered.empty:
         st.warning(f"No valid data points after dropping NaNs for average '{value_label}' by '{category_label}' plot.")
         return

    print(f"Generating Average {value_label} by {category_label} bar chart.")
    avg_value_by_category = df_filtered.groupby(category_col)[value_col].mean().reset_index()

    fig = px.bar(
        avg_value_by_category,
        x=category_col,
        y=value_col,
        title=f'Average {value_label} by {category_label}',
        labels={category_col: category_label, value_col: value_label} # Set labels
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_optimal_treatment_breakdown(df: pd.DataFrame, optimal_treatment_col: str, optimal_treatment_label: str):
    """Plots the count or proportion of customers for each optimal treatment."""
    if df is None or df.empty or optimal_treatment_col not in df.columns or df[optimal_treatment_col].isnull().all():
        st.warning(f"No valid data in '{optimal_treatment_col}' column for breakdown plot.")
        return

    print(f"Generating breakdown plot for {optimal_treatment_label}.")

    treatment_counts = df[optimal_treatment_col].value_counts().reset_index()
    treatment_counts.columns = [optimal_treatment_col, 'Count']

    treatment_counts = treatment_counts.sort_values('Count', ascending=False)

    fig = px.bar(
        treatment_counts,
        x=optimal_treatment_col,
        y='Count',
        title=f'Number of Customers by {optimal_treatment_label}',
        labels={optimal_treatment_col: optimal_treatment_label, 'Count': 'Number of Customers'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_value_vs_original_feature_colored(df: pd.DataFrame, original_feature_col: str, value_col: str, optimal_treatment_col: str, original_feature_label: str, value_label: str, optimal_treatment_label: str):
     """Plots a value column vs. an original feature column, colored by optimal treatment."""
     required_cols = [value_col, original_feature_col, optimal_treatment_col]
     if df is None or df.empty or not all(col in df.columns for col in required_cols):
         st.warning(f"Data or required columns ({', '.join(required_cols)}) not found for {value_label} vs {original_feature_label} colored plot.")
         return
     # Drop rows with NaNs in key plotting columns
     df_filtered = df.dropna(subset=[value_col, original_feature_col, optimal_treatment_col])
     if df_filtered.empty:
          st.warning(f"No valid data points after dropping NaNs for {value_label} vs {original_feature_label} colored plot.")
          return

     print(f"Generating {value_label} vs {original_feature_label} scatter plot colored by {optimal_treatment_label}.")

     fig = px.scatter(
         df_filtered,
         x=original_feature_col,
         y=value_col,
         color=optimal_treatment_col,
         labels={
             original_feature_col: original_feature_label,
             value_col: value_label,
             optimal_treatment_col: optimal_treatment_label
             },
         hover_data=['customer_id', value_col, original_feature_col, optimal_treatment_col] if 'customer_id' in df_filtered.columns else [value_col, original_feature_col, optimal_treatment_col],
         title=f'{value_label} vs. {original_feature_label} colored by {optimal_treatment_label}'
     )
     st.plotly_chart(fig, use_container_width=True)