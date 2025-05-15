# model_handler.py

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, OUTCOME_LABELS


@st.cache_resource
def load_ml_model(model_path: str):
    """Loads the trained machine learning model pipeline."""
    try:
        # Load the entire pipeline object
        pipeline = joblib.load(model_path)
        print("ML model pipeline loaded successfully.")

        if not isinstance(pipeline, Pipeline):
             st.error("Loaded object is not a scikit-learn Pipeline.")
             return None

        if 'preprocessor' not in pipeline.named_steps or not isinstance(pipeline.named_steps['preprocessor'], ColumnTransformer):
             st.error("Loaded pipeline does not contain a 'preprocessor' step which is a ColumnTransformer.")
             return None

        classifier = pipeline.named_steps.get('classifier')
        if classifier is None:
             st.error("Loaded pipeline does not contain a 'classifier' step.")
             return None

        if not hasattr(classifier, 'predict_proba'):
             st.error("Loaded classifier does not have a predict_proba method.")
             return None

        model_classes = list(classifier.classes_)
        if sorted(model_classes) != sorted(OUTCOME_LABELS):
             st.warning(f"Model classes {model_classes} do not match expected outcome labels {OUTCOME_LABELS}. Predictions might be misinterpreted.")
        return pipeline

    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Please check the path in config.py.")
        print(f"Model file not found: {MODEL_PATH}")
        return None
    except Exception as e:
        st.error(f"Error loading ML model pipeline: {e}")
        print(f"Error loading ML model pipeline: {e}")
        return None


def predict_probabilities(pipeline: Pipeline, raw_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Makes probability predictions using the loaded model pipeline.
    The input `raw_data_before_engineering` should be the DataFrame
    *after* initial feature engineering steps (like binary conversion,
    discount_change, renewal_month extraction) but *before* the ColumnTransformer.
    The pipeline handles the ColumnTransformer application internally.
    """
    if pipeline is None:
        st.error("ML model pipeline is not loaded.")
        return None
    if raw_df is None or raw_df.empty:
         st.warning("No data provided for prediction.")
         return pd.DataFrame()

    try:

        probabilities = pipeline.predict_proba(raw_df)

        classifier = pipeline.named_steps['classifier']
        class_labels = classifier.classes_
        prob_df = pd.DataFrame(probabilities, columns=[f'prob_{label}' for label in class_labels])
        for label in OUTCOME_LABELS:
            if f'prob_{label}' not in prob_df.columns:
                prob_df[f'prob_{label}'] = 0.0 #if missing anything

        prob_df = prob_df[[f'prob_{label}' for label in OUTCOME_LABELS]]
        print("Predictions made successfully using the pipeline.")
        return prob_df

    except Exception as e:
        st.error(f"Error during pipeline prediction: {e}")
        print(f"Error during pipeline prediction: {e}")
        return None

