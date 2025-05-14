# visualization_handler.py

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from config import OUTCOME_LABELS

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

def plot_expected_outcome_counts_comparison(expected_counts_df: pd.DataFrame):
    """Plots a grouped bar chart comparing expected customer counts per outcome by strategy."""
    if expected_counts_df is None or expected_counts_df.empty:
        st.warning("No data for expected outcome counts comparison plot.")
        return

    print("Generating expected outcome counts comparison bar chart.")

    fig = px.bar(
        expected_counts_df,
        x='Outcome',
        y='Expected Customers',
        color='Strategy',
        barmode='group',
        title='Expected Number of Customers per Outcome by Strategy',
        labels={'Outcome': 'Predicted Outcome', 'Expected Customers': 'Expected Number of Customers', 'Strategy': 'Optimization Strategy'}
    )
    st.plotly_chart(fig, use_container_width=True)

