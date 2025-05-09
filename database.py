import streamlit as st
import pandas as pd

# Use 'df_original_full' consistently as the key for the main dataframe
MAIN_DF_SESSION_STATE_KEY = 'df_original_full'

def save_dataframe_to_session_state(df: pd.DataFrame):
    """Saves a pandas DataFrame to Streamlit's session state."""
    st.session_state[MAIN_DF_SESSION_STATE_KEY] = df
    print("Data saved to session state under key:", MAIN_DF_SESSION_STATE_KEY)

def load_dataframe_from_session_state() -> pd.DataFrame | None:
    """Loads data from Streamlit's session state."""
    df = st.session_state.get(MAIN_DF_SESSION_STATE_KEY)
    if df is not None:
        print("Data loaded from session state using key:", MAIN_DF_SESSION_STATE_KEY)
    return df

def clear_session_state_data():
    """Clears the data from session state."""
    if MAIN_DF_SESSION_STATE_KEY in st.session_state:
        del st.session_state[MAIN_DF_SESSION_STATE_KEY]
        print("Data cleared from session state using key:", MAIN_DF_SESSION_STATE_KEY)
    # Note: In app.py, we will clear other related keys explicitly