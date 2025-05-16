# model_handler.py

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import MODEL_PATH, OUTCOME_LABELS


@st.cache_resource
def load_ml_model(model_path: str):
    """Loads the trained machine learning model pipeline and attaches class label mapping."""
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

        num_to_label = {cls: label for cls, label in zip(sorted(model_classes), OUTCOME_LABELS)}

        pipeline.class_label_mapping = num_to_label

        if sorted(model_classes) != sorted(num_to_label.keys()):
            st.warning(f"Model classes {model_classes} do not match expected numeric classes {list(num_to_label.keys())}. Predictions might be misinterpreted.")

        return pipeline

    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please check the path.")
        print(f"Model file not found: {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading ML model pipeline: {e}")
        print(f"Error loading ML model pipeline: {e}")
        return None



def predict_probabilities(pipeline: Pipeline, raw_df: pd.DataFrame) -> pd.DataFrame | None:
    if pipeline is None:
        st.error("ML model pipeline is not loaded.")
        return None
    if raw_df is None or raw_df.empty:
        st.warning("No data provided for prediction.")
        return pd.DataFrame()


    try:
        invalid_cols = raw_df.select_dtypes(exclude=["number", "category"]).columns.tolist()
        if invalid_cols:
            print("Invalid columns (not numeric or categorical):", invalid_cols)
            raise ValueError(f"Input DataFrame contains non-numeric or non-categorical object types: {invalid_cols}")



        try:
            probabilities = pipeline.predict_proba(raw_df)
        except Exception as e:
            st.error(" Error in batch prediction. Trying row-by-row...")

        classifier = pipeline.named_steps['classifier']
        # Your model's classes might be:
        model_classes = list(classifier.classes_)  # e.g., [0, 1, 2]

        # Create a mapping from model classes to your labels
        # You must know the correct mapping order here; adjust as needed
        class_label_mapping = {0: 'Accepted', 1: 'Call_Back', 2: 'Churn'}

        # Build the probabilities DataFrame with model classes first
        prob_df = pd.DataFrame(probabilities, columns=[f'prob_{cls}' for cls in model_classes])

        # Rename columns from numeric class to your labels
        prob_df.rename(columns={f'prob_{cls}': f'prob_{class_label_mapping[cls]}' for cls in model_classes},
                       inplace=True)

        # Ensure all expected outcome labels are present, fill missing with 0.0
        for label in OUTCOME_LABELS:
            if f'prob_{label}' not in prob_df.columns:
                prob_df[f'prob_{label}'] = 0.0

        # Reorder columns according to OUTCOME_LABELS
        prob_df = prob_df[[f'prob_{label}' for label in OUTCOME_LABELS]]

        print("✅ Predictions made successfully.")
        return prob_df

    except Exception as e:
        st.error(f"Error during pipeline prediction: {e}")
        print(f"🔥 Error during pipeline prediction: {e}")
        return None
