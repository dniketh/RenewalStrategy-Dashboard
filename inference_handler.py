import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from data_handler import preprocess_dataframe
from config import TREATMENT_INPUT_COL, DISCOUNT_INPUT_COL, OUTCOME_LABELS
from model_handler import predict_probabilities


def compute_expected_value(
    prob_dict: dict,
    outcome_values: dict
) -> float:
    """
    Computes the expected value for a single data point
    based on predicted probabilities and predefined outcome values.
    """
    ev = 0.0
    for outcome, prob in prob_dict.items():
        if outcome in outcome_values:
            ev += prob * outcome_values[outcome]
    return ev


def evaluate_scenarios_per_customer(
    df: pd.DataFrame,
    possible_treatments: list,
    user_discount: float,
    model_pipeline: Pipeline,
    outcome_values: dict,
    retention_class_label: str
) -> pd.DataFrame:
    """
    Evaluates all possible treatments for each customer in the DataFrame sample
    to find the optimal treatment for maximizing EV and Retention Probability.
    Stores probabilities for both optimal treatment scenarios.
    Returns a DataFrame with per-customer optimal results and relevant original features.
    """
    results_list = []
    total_customers_in_sample = len(df)
    progress_bar = st.progress(0, text="Evaluating customer scenarios...")
    original_features_to_include = ['customer_id', 'usage', 'state', 'cust_tenure', 'before_discount']

    for i, (index, customer_row) in enumerate(df.iterrows()):
        customer_id = customer_row.get('customer_id', f'Row_{index}')

        customer_best_ev = -np.inf
        customer_optimal_ev_treatment = None
        customer_best_prob_accepts = -np.inf
        customer_optimal_prob_treatment = None

        optimal_ev_prob_dict = None #For EV
        optimal_prob_prob_dict = None # For Customer Retention


        for treatment in possible_treatments:

            scenario_df = pd.DataFrame([customer_row.copy()])
            if treatment == "Lower Discount" and user_discount < scenario_df[DISCOUNT_INPUT_COL].item():
                scenario_df[DISCOUNT_INPUT_COL] = user_discount
            elif treatment == "Lower Discount" and user_discount >= scenario_df[DISCOUNT_INPUT_COL].item():
                st.warning("Skipping Lower Discount Treatment for this customer as discount offered is higher than previous contract.")
                continue
            elif treatment in ["Same Contract", "Remove Discount"]:
                 scenario_df[DISCOUNT_INPUT_COL] = 0.0


            scenario_df[TREATMENT_INPUT_COL] = treatment
            try:
                df_engineered_scenario = preprocess_dataframe(scenario_df)
            except Exception as e:
                 st.warning(f"Feature engineering failed for customer {customer_id}, treatment '{treatment}': {e}. Skipping this scenario.")
                 df_engineered_scenario = None

            if df_engineered_scenario is None or df_engineered_scenario.empty:
                 continue
            try:
                prob_df_scenario = predict_probabilities(model_pipeline, df_engineered_scenario)
            except Exception as e:
                 st.warning(f"Prediction failed for customer {customer_id}, treatment '{treatment}': {e}. Skipping this scenario.")
                 prob_df_scenario = None
            if prob_df_scenario is None or prob_df_scenario.empty:
                 continue
            prob_dict = {label: prob_df_scenario.iloc[0].get(f'prob_{label}', np.nan) for label in OUTCOME_LABELS}
            current_ev = compute_expected_value(prob_dict, outcome_values)

            current_prob_accepts = prob_dict.get(retention_class_label, 0.0) # Use the label directly from prob_dict

            if current_ev > customer_best_ev:
                customer_best_ev = current_ev
                customer_optimal_ev_treatment = treatment
                optimal_ev_prob_dict = prob_dict.copy()


            if current_prob_accepts > customer_best_prob_accepts:
                customer_best_prob_accepts = current_prob_accepts
                customer_optimal_prob_treatment = treatment
                optimal_prob_prob_dict = prob_dict.copy()

        result_row = {'customer_id': customer_id}
        for col in original_features_to_include:
             if col in customer_row.index:
                  result_row[col] = customer_row[col]
             else:
                  result_row[col] = np.nan


        result_row.update({
            'Maximum Expected Value': customer_best_ev if customer_best_ev > -np.inf else np.nan, # Handle case where no scenario worked
            'Best Treatment Plan for Maximum Value': customer_optimal_ev_treatment,
            'Maximum Probability of Accepting Contract with Optimal Treatment': customer_best_prob_accepts if customer_best_prob_accepts > -np.inf else np.nan, # Handle case where no scenario worked
            'Best Treatment Plan for Accepting Renewal': customer_optimal_prob_treatment,
            **{f'Probability of Outcome {label} with Maximum Expected Value': optimal_ev_prob_dict.get(label, np.nan) if optimal_ev_prob_dict else np.nan
               for label in OUTCOME_LABELS},
            **{f'Probability of Outcome {label} with Maximum Likelihood of Accepting Renewal': optimal_prob_prob_dict.get(label, np.nan) if optimal_prob_prob_dict else np.nan
               for label in OUTCOME_LABELS}
        })
        results_list.append(result_row)


        progress = (i + 1) / total_customers_in_sample
        progress_bar.progress(progress, text=f"Evaluating customer scenarios and treatment plans... {i+1}/{total_customers_in_sample}")

    progress_bar.empty()


    results_df = pd.DataFrame(results_list)

    return results_df

