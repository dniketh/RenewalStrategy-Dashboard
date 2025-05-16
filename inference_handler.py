import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from data_handler import preprocess_dataframe
from config import TREATMENT_INPUT_COL, DISCOUNT_INPUT_COL, OUTCOME_LABELS
from model_handler import predict_probabilities



def compute_historical_ev(row, actual_outcome) -> float:
    """
    Compute the historical expected value from the actual data row.
    Uses the same business logic as compute_expected_value but for the observed actual outcome.
    """
    treatment = row.get(TREATMENT_INPUT_COL) or row.get('actual_treatment')  # adapt 'actual_treatment' if different
    usage = row.get('usage', 0.0)
    before_discount = row.get('before_discount', 0.0)


    # Constants (same as compute_expected_value)
    v_rev = 247.0
    f_rev = 436.0
    v_cost = 209.0
    f_cost = 269.0

    baseline_value = (1 - before_discount) * (usage * v_rev + f_rev) - (usage * v_cost + f_cost)
    value_churn = -baseline_value
    value_call_back = 0.0

    if treatment == 'Same Contract':
        value_accepts = 0
    elif treatment == 'Remove Discount':
        value_accepts = before_discount * (usage * v_rev + f_rev)
    else:
        # Use the discount column from the row if available
        discount_offered = row.get(DISCOUNT_INPUT_COL, 0.0)
        value_accepts = (before_discount - discount_offered) * (usage * v_rev + f_rev)

    dynamic_outcome_values = {

        'Accepted': value_accepts,

        'Call_Back': value_call_back,
        'Churn': value_churn
    }
    if actual_outcome.item() not in dynamic_outcome_values:
        print(f"Unrecognized outcome: {actual_outcome}")

    return dynamic_outcome_values.get(actual_outcome.item(), np.nan)


def compute_expected_value(
    prob_dict: dict,
    treatment: str,
    usage: float,
    scenario_discount_offered: float,
    before_discount
) -> float:
    """
    Computes the expected value for a single data point
    based on predicted probabilities and predefined outcome values.
    """
    ev = 0.0
    value_accepts = 0.0
    value_call_back = 0.0
    value_churn = 0.0

    v_rev = 247.0 # Variable revenue per MWh
    f_rev = 436.0 # Fixed revenue per year
    v_cost = 209.0 # Variable cost per MWh
    f_cost = 269.0 # Fixed cost per year

    baseline_value = (1-before_discount) * (usage * v_rev + f_rev) - (usage * v_cost + f_cost)
    value_churn = -baseline_value
    value_call_back = 0.0

    if treatment == 'Same Contract':
        value_accepts = 0
    elif treatment == 'Remove Discount':
        value_accepts = before_discount * (usage * v_rev + f_rev)
    else:
        value_accepts = (before_discount - scenario_discount_offered) * (usage * v_rev + f_rev)

    dynamic_outcome_values = {
        'Accepted': value_accepts,
        'Call_Back': value_call_back,
        'Churn': value_churn
    }

    for outcome, prob in prob_dict.items():
        if pd.notna(prob) and outcome in dynamic_outcome_values:
            ev += prob * dynamic_outcome_values[outcome]
    return float(ev)


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
    skip_customer_flag = False
    actual_outcome = None
    for i, (index, customer_row) in enumerate(df.iterrows()):
        customer_id = customer_row.get('customer_id', f'Row_{index}')


        customer_best_ev = -np.inf #For EV
        customer_optimal_ev_treatment = None #For EV
        ev_optimized_accept_prob = 0.0 #For EV
        customer_best_prob_accepts = -np.inf # For Customer Retention
        customer_optimal_prob_treatment = None # For Customer Retention
        retention_optimized_ev = -np.inf # For Customer Retention
        optimal_ev_prob_dict = None #For EV
        optimal_prob_prob_dict = None # For Customer Retention

        scenario_df = pd.DataFrame([customer_row.copy()])
        for treatment in possible_treatments:

            actual_outcome = scenario_df['renewal_outcome']
            if treatment == "Lower Discount" and user_discount < scenario_df[DISCOUNT_INPUT_COL].item():
                scenario_df[DISCOUNT_INPUT_COL] = user_discount
            elif treatment == "Lower Discount" and user_discount >= scenario_df[DISCOUNT_INPUT_COL].item():
                skip_customer_flag = True
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
            current_ev = compute_expected_value(prob_dict, treatment, scenario_df['usage'], user_discount, scenario_df['before_discount'])

            current_prob_accepts = prob_dict.get(retention_class_label, 0.0)

            if current_ev > customer_best_ev: #for ev
                customer_best_ev = current_ev
                customer_optimal_ev_treatment = treatment
                optimal_ev_prob_dict = prob_dict.copy()
                ev_optimized_accept_prob = current_prob_accepts


            if current_prob_accepts > customer_best_prob_accepts: #for retention
                customer_best_prob_accepts = current_prob_accepts
                retention_optimized_ev = current_ev
                customer_optimal_prob_treatment = treatment
                optimal_prob_prob_dict = prob_dict.copy()

        result_row = {'customer_id': customer_id}
        for col in original_features_to_include:
             if col in customer_row.index:
                  result_row[col] = customer_row[col]
             else:
                  result_row[col] = np.nan

        result_row.update({
            'Maximum Expected Value': customer_best_ev if customer_best_ev > -np.inf else np.nan,
            'Best Treatment Plan for Maximum Value': customer_optimal_ev_treatment,
            'Retention-Optimized Accept Prob': customer_best_prob_accepts if customer_best_prob_accepts > -np.inf else np.nan,
            'Best Treatment Plan for Accepting Renewal': customer_optimal_prob_treatment,
            'Maximum Expected Value (For Retention)': retention_optimized_ev,
            'EV Optimized Accept Probability': ev_optimized_accept_prob if pd.notna(
                ev_optimized_accept_prob) else np.nan,
            **{f'Probability of Outcome {label} with Maximum Expected Value': optimal_ev_prob_dict.get(label,
                                                                                                       np.nan) if optimal_ev_prob_dict else np.nan
               for label in OUTCOME_LABELS},
            **{
                f'Probability of Outcome {label} with Maximum Likelihood of Accepting Renewal': optimal_prob_prob_dict.get(
                    label, np.nan) if optimal_prob_prob_dict else np.nan
                for label in OUTCOME_LABELS}
        })

        historical_ev = compute_historical_ev(customer_row, actual_outcome)
        result_row['Historical Expected Value'] = historical_ev

        results_list.append(result_row)




        progress = (i + 1) / total_customers_in_sample
        progress_bar.progress(progress, text=f"Evaluating customer scenarios and treatment plans... {i+1}/{total_customers_in_sample}")

    progress_bar.empty()

    if skip_customer_flag is True:
        st.warning("Skipped some customers for lower treatment plan due to discount being offered is less than previous contract..  ")
    results_df = pd.DataFrame(results_list)

    print("===============Results===================")
    print(results_df)
    results_df.to_excel('Results.xlsx')
    return results_df

