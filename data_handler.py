import numpy as np
import pandas as pd
import streamlit as st

from config import COLUMNS_TO_DROP_RAW


def load_csv(uploaded_file) -> pd.DataFrame | None:
    """Loads data from an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        print("CSV loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        print(f"Error loading CSV file: {e}")
        return None



def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    df_processed = df.copy()

    binary_columns = ['green', 'dual_fuel_customer', 'direct_debit_flag']
    for col in binary_columns:
        if col in df_processed.columns and df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})

    categorical_columns = ['state', 'communication_preference', 'age',
                           'before_channel', 'treatment_given']
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown').astype('category')

    if 'before_discount' in df_processed.columns and 'discount_offered' in df_processed.columns:
        df_processed['discount_change'] = df_processed['discount_offered'] - df_processed['before_discount']

    if 'renewal_date' in df_processed.columns:
        try:
            df_processed['renewal_month'] = pd.to_datetime(df_processed['renewal_date']).dt.month.astype('category')
        except Exception:
            print("Warning: Could not parse renewal_date.")

    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(-1)

    df_processed = df_processed.drop(columns=['customer_id', 'renewal_date', 'renewal_outcome'], errors='ignore')

    return df_processed

