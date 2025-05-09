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
    """
    Applies initial feature engineering steps to the DataFrame.
    The input df can be a single row or multiple rows (from the sampled data).
    Returns the DataFrame after engineering, before the ColumnTransformer.
    """
    df_engineered = df.copy()
    binary_columns = ['green', 'dual_fuel_customer', 'direct_debit_flag']
    for col in binary_columns:
        if col in df_engineered.columns:
            df_engineered.loc[:, col] = df_engineered[col].map({'Yes': 1, 'No': 0})
            df_engineered.loc[:, col] = pd.to_numeric(df_engineered[col], errors='coerce')

    if 'before_discount' in df_engineered.columns and 'discount_offered' in df_engineered.columns:
         df_engineered.loc[:, 'before_discount'] = pd.to_numeric(df_engineered['before_discount'], errors='coerce')
         df_engineered.loc[:, 'discount_offered'] = pd.to_numeric(df_engineered['discount_offered'], errors='coerce')
         df_engineered.loc[:, 'discount_change'] = df_engineered['discount_offered'] - df_engineered['before_discount']
    else:
         df_engineered.loc[:, 'discount_change'] = np.nan

    if 'renewal_date' in df_engineered.columns:
        try:
            df_engineered.loc[:, 'renewal_date_dt'] = pd.to_datetime(df_engineered['renewal_date'], errors='coerce')
            df_engineered.loc[:, 'renewal_month'] = df_engineered['renewal_date_dt'].dt.month
            df_engineered.loc[:, 'renewal_month'] = df_engineered['renewal_month'].astype('category')
        except Exception as e:
            print(f"Could not convert renewal_date to datetime or extract month: {e}. Adding 'renewal_month' with NaNs.")
            df_engineered.loc[:, 'renewal_month'] = np.nan
            df_engineered.loc[:, 'renewal_month'] = df_engineered['renewal_month'].astype('category')

    columns_to_drop_engineered = COLUMNS_TO_DROP_RAW + ['renewal_date_dt']
    df_engineered = df_engineered.drop(columns=columns_to_drop_engineered, errors='ignore')



    print(f"Feature engineering applied successfully. DataFrame shape: {df_engineered.shape}")
    return df_engineered

