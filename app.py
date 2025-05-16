import pandas as pd
import streamlit as st

from config import (
    MODEL_PATH, DISCOUNT_RANGE, OUTCOME_VALUES, OUTCOME_LABELS, RETENTION_CLASS_LABEL,
    POSSIBLE_TREATMENTS,
    STRATEGY_OPTIONS, STRATEGY_RETENTION, STRATEGY_EXPECTED_VALUE, STRATEGY_COMPARISON,
    DEFAULT_CUSTOMER_LIMIT, RANDOM_STATE_SAMPLING
)
from data_handler import load_csv
from database import MAIN_DF_SESSION_STATE_KEY, load_dataframe_from_session_state
from inference_handler import \
    evaluate_scenarios_per_customer
from model_handler import load_ml_model
from visualization_handler import (
    plot_optimal_treatment_breakdown,
    plot_expected_outcome_counts_comparison,
    plot_expected_outcome_pie_chart_comparison, plot_single_strategy_expected_counts
)
st.set_page_config(
    page_title="AI-Driven Contract Renewal Optimization System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI-Driven Contract Renewal Optimization System")

if 'uploaded_data_processed' not in st.session_state:
    st.session_state['uploaded_data_processed'] = False
if 'df_per_customer_results' not in st.session_state:
    st.session_state['df_per_customer_results'] = None
if 'calculate_button_clicked' not in st.session_state:
    st.session_state['calculate_button_clicked'] = False
if 'selected_strategy' not in st.session_state:
    st.session_state['selected_strategy'] = STRATEGY_EXPECTED_VALUE # Default to EV
if 'processed_customer_count_input' not in st.session_state:
     st.session_state['processed_customer_count_input'] = DEFAULT_CUSTOMER_LIMIT
if 'actual_processed_count' not in st.session_state:
    st.session_state['actual_processed_count'] = 0

if 'just_cleared' not in st.session_state:
     st.session_state['just_cleared'] = False

if st.session_state.get('just_cleared', False):
    st.session_state['just_cleared'] = False

with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])

    if uploaded_file is not None and not st.session_state.get('just_cleared', False):
        df = load_csv(uploaded_file)
        if df is not None:
            st.session_state[MAIN_DF_SESSION_STATE_KEY] = df

            st.session_state['uploaded_data_processed'] = True
            st.session_state['df_per_customer_results'] = None
            st.session_state['calculate_button_clicked'] = False
            st.session_state['actual_processed_count'] = 0
            st.success(f"CSV uploaded successfully. Contains {len(df)} rows.")
            st.session_state['processed_customer_count_input'] = min(len(df), DEFAULT_CUSTOMER_LIMIT)

    st.header("Choose Strategy")
    selected_strategy = st.radio(
        "Select Optimization Goal:",
        STRATEGY_OPTIONS, # Use the updated list from config
        index=STRATEGY_OPTIONS.index(st.session_state.get('selected_strategy', STRATEGY_EXPECTED_VALUE)), # Maintain selection across reruns
        key='strategy_radio'
    )
    st.session_state['selected_strategy'] = selected_strategy


    st.header("Simulation Settings")
    df_original = load_dataframe_from_session_state()

    user_discount = None

    if df_original is not None:
       total_customers = len(df_original)
       st.info(f"Total customers in uploaded data: {total_customers}")

       customer_limit_input = st.number_input(
           "Limit number of customers to process:",
           min_value=1,
           max_value=total_customers,
           value=st.session_state.get('processed_customer_count_input', DEFAULT_CUSTOMER_LIMIT),
           step=1,
           help="Process a subset of customers to manage calculation time. Uses random sampling.",
           key='customer_limit_input'
       )
       st.session_state['processed_customer_count_input'] = customer_limit_input


       user_discount = st.slider(
           f"Set Discount for '{POSSIBLE_TREATMENTS[2]}':",
           min_value=DISCOUNT_RANGE[0],
           max_value=DISCOUNT_RANGE[1],
           value=st.session_state.get('discount_slider', DISCOUNT_RANGE[0]),
           step=0.01,
           format="%.2f",
           key='discount_slider',
           help="This discount is applied when evaluating the 'Lower Discount' treatment scenario."
       )
    else:
       st.info("Upload data first to set simulation settings.")


    st.header("Actions")
    if st.session_state.get(MAIN_DF_SESSION_STATE_KEY) is not None and user_discount is not None:
        if st.button("Run Simulation & Calculate Results"):
            st.session_state['calculate_button_clicked'] = True

        if st.button("Clear Uploaded Data"):
            st.session_state['just_cleared'] = True

            keys_to_clear = [
                MAIN_DF_SESSION_STATE_KEY,  # The main uploaded data DataFrame
                'uploaded_data_processed',  # Flag indicating if data has been processed (uploaded)
                'df_per_customer_results',  # Stored calculation results DataFrame
                'calculate_button_clicked', # Flag for triggering the calculation
                'actual_processed_count',   # Actual count of customers processed in the last run
                'processed_customer_count_input', # The value from the customer limit number input
                'selected_strategy',        # The value from the strategy radio button
                'strategy_radio',
                'customer_limit_input',
                'discount_slider'
            ]

            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state['selected_strategy'] = STRATEGY_EXPECTED_VALUE
            st.session_state['processed_customer_count_input'] = DEFAULT_CUSTOMER_LIMIT


            st.success("Data and results cleared. Please upload a new file.")
            st.rerun()
    elif st.session_state.get(MAIN_DF_SESSION_STATE_KEY) is not None:
         st.warning("Cannot calculate without setting simulation inputs.")
    else:
         st.info("Upload data to enable simulations.")



df_original = load_dataframe_from_session_state()

if st.session_state.get('calculate_button_clicked', False) and df_original is not None and user_discount is not None and not st.session_state.get('just_cleared', False):
    st.subheader("Running Simulation...")

    customer_limit = st.session_state['processed_customer_count_input']
    actual_limit = min(len(df_original), customer_limit)

    df_sample = df_original.sample(n=actual_limit, random_state=RANDOM_STATE_SAMPLING).reset_index(drop=True)
    st.session_state['actual_processed_count'] = len(df_sample)
    st.info(f"Processing {len(df_sample)} customers...")


    with st.spinner("Loading ML model..."):
        model_pipeline = load_ml_model(MODEL_PATH)
        if model_pipeline is None:
            st.error("Model loading failed. Cannot proceed.")
            st.session_state['calculate_button_clicked'] = False
            st.stop()
        st.success("Model loaded.")

    with st.spinner(f"Evaluating {len(POSSIBLE_TREATMENTS)} scenarios for each of the {len(df_sample)} customers..."):

        df_per_customer_results = evaluate_scenarios_per_customer(
             df_sample,
             POSSIBLE_TREATMENTS,
             user_discount,
             model_pipeline,
             OUTCOME_VALUES,
             RETENTION_CLASS_LABEL,
         )

        st.session_state['df_per_customer_results'] = df_per_customer_results
        st.success("Simulation complete.")

    st.session_state['calculate_button_clicked'] = False


if st.session_state['df_per_customer_results'] is not None and not st.session_state['df_per_customer_results'].empty and not st.session_state.get('just_cleared', False):

    processed_count = st.session_state['actual_processed_count']
    st.subheader(f"Simulation Results for {processed_count} Customers")


    df_results = st.session_state['df_per_customer_results']
    st.subheader("Overall Metrics")

    if 'Maximum Expected Value' in df_results.columns and not df_results['Maximum Expected Value'].isnull().all():
        total_expected_value = df_results['Maximum Expected Value'].sum()
        st.metric(label="Total Expected Value (Optimal EV Treatment)", value=f"${total_expected_value:,.2f}") # Format as currency
    else:
         st.warning("Total Expected Value (Optimal EV Treatment) could not be computed.")

    if 'Maximum Expected Value (For Retention)' in df_results.columns and not df_results['Maximum Expected Value (For Retention)'].isnull().all():
        total_expected_value_retention = df_results['Maximum Expected Value (For Retention)'].sum()
        st.metric(label="Total Expected Value (Optimal Retention Treatment)", value=f"${total_expected_value_retention:,.2f}") # Format as currency
    else:
         st.warning("Total Expected Value (Optimal Retention Treatment) could not be computed.")

   # if 'Historical Expected Value' in df_results.columns and not df_results['Historical Expected Value'].isnull().all():
        #total_historical_value  = df_results['Historical Expected Value'].sum()
        #st.metric(label="Total Historical Value (Optimal Retention Treatment)", value=f"${total_historical_value:,.2f}") # Format as currency

    st.subheader("Expected Churn Comparison")

    churn_outcome_label = 'Churn'
    expected_churn_data = []
    total_customers = st.session_state.get('actual_processed_count', 0)

    ev_churn_col = f'Probability of Outcome {churn_outcome_label} with Maximum Expected Value'
    if ev_churn_col in df_results.columns and total_customers > 0:
        expected_churn_count_ev = df_results[ev_churn_col].sum()
        expected_churn_percentage_ev = (expected_churn_count_ev / total_customers) * 100
        expected_churn_data.append({
            'Strategy': STRATEGY_EXPECTED_VALUE,
            'Expected Churn Count': expected_churn_count_ev,
            'Expected Churn Percentage': expected_churn_percentage_ev
        })
    else:
        st.warning(f"Expected Churn data for '{STRATEGY_EXPECTED_VALUE}' not available.")


    # Expected Churn for Optimal Retention Strategy
    retention_churn_col = f'Probability of Outcome {churn_outcome_label} with Maximum Likelihood of Accepting Renewal'
    if retention_churn_col in df_results.columns and total_customers > 0:
        expected_churn_count_retention = df_results[retention_churn_col].sum()
        expected_churn_percentage_retention = (expected_churn_count_retention / total_customers) * 100
        expected_churn_data.append({
            'Strategy': STRATEGY_RETENTION,
            'Expected Churn Count': expected_churn_count_retention,
            'Expected Churn Percentage': expected_churn_percentage_retention
        })
    else:
        st.warning(f"Expected Churn data for '{STRATEGY_RETENTION}' not available.")

    if expected_churn_data:
        expected_churn_df = pd.DataFrame(expected_churn_data)
        expected_churn_df['Expected Churn Percentage'] = expected_churn_df['Expected Churn Percentage'].map('{:.2f}%'.format)
        st.dataframe(expected_churn_df, hide_index=True)
    else:
        st.info("Could not generate Expected Churn Comparison table.")


    selected_strategy = st.session_state['selected_strategy']

    if selected_strategy == STRATEGY_COMPARISON:
        st.subheader("Expected Customer Distribution per Outcome by Strategy")

        expected_counts_list = []

        ev_prob_cols = [f'Probability of Outcome {label} with Maximum Expected Value' for label in OUTCOME_LABELS]
        if all(col in df_results.columns for col in ev_prob_cols):
            for label in OUTCOME_LABELS:
                expected_counts_list.append({
                    'Outcome': label,
                    'Expected Customers': df_results[f'Probability of Outcome {label} with Maximum Expected Value'].sum(),
                    'Strategy': STRATEGY_EXPECTED_VALUE
                })
        else:
             st.warning("Probability columns for Optimal EV Strategy not found for comparison plot.")


        # Data for Optimal Retention Strategy
        prob_prob_cols = [f'Probability of Outcome {label} with Maximum Likelihood of Accepting Renewal' for label in OUTCOME_LABELS]
        # Ensure all required probability columns exist before summing
        if all(col in df_results.columns for col in prob_prob_cols):
            for label in OUTCOME_LABELS:
                expected_counts_list.append({
                    'Outcome': label,
                    'Expected Customers': df_results[f'Probability of Outcome {label} with Maximum Likelihood of Accepting Renewal'].sum(),
                    'Strategy': STRATEGY_RETENTION
                })
        else:
             st.warning("Probability columns for Optimal Retention Strategy not found for comparison plot.")


        expected_counts_comparison_df = pd.DataFrame(expected_counts_list)
        if not expected_counts_comparison_df.empty:
            plot_expected_outcome_pie_chart_comparison(expected_counts_comparison_df)
            plot_expected_outcome_counts_comparison(expected_counts_comparison_df)
        else:
            st.info("Could not generate expected outcome counts comparison plot.")

        st.subheader("Optimal Treatment Breakdown by Strategy")

        optimal_treatment_ev_col = 'Best Treatment Plan for Maximum Value'
        optimal_treatment_ev_label = 'Optimal Treatment (Maximize Value)'
        if optimal_treatment_ev_col in df_results.columns:
             st.write(f"**{optimal_treatment_ev_label}**")
             plot_optimal_treatment_breakdown(df_results, optimal_treatment_ev_col, optimal_treatment_ev_label)
        else:
             st.warning(f"Column '{optimal_treatment_ev_col}' not found for breakdown plot.")


        optimal_treatment_prob_col = 'Best Treatment Plan for Accepting Renewal'
        optimal_treatment_prob_label = 'Optimal Treatment (Maximize Retention)'
        if optimal_treatment_prob_col in df_results.columns:
             st.write(f"**{optimal_treatment_prob_label}**")
             plot_optimal_treatment_breakdown(df_results, optimal_treatment_prob_col, optimal_treatment_prob_label)
        else:
             st.warning(f"Column '{optimal_treatment_prob_col}' not found for breakdown plot.")


    else:
        st.info(f"Showing results optimized for: **{selected_strategy}**")

        if selected_strategy == STRATEGY_RETENTION:
            value_col = 'Retention-Optimized Accept Prob'
            value_label = f'Maximum Predicted Probability of {RETENTION_CLASS_LABEL}'
            optimal_treatment_col = 'Best Treatment Plan for Accepting Renewal'
            optimal_treatment_label = 'Optimal Treatment (Maximize Retention)'
            prob_cols_prefix = 'Probability of Outcome {label} with Maximum Likelihood of Accepting Renewal'
        else:
            value_col = 'Maximum Expected Value'
            value_label = 'Maximum Expected Value'
            optimal_treatment_col = 'Best Treatment Plan for Maximum Value'
            optimal_treatment_label = 'Optimal Treatment (Maximize Value)'
            prob_cols_prefix = 'Probability of Outcome {label} with Maximum Expected Value'


        if value_col in df_results.columns and optimal_treatment_col in df_results.columns:

            st.subheader(f"Expected Customer Counts per Outcome ({selected_strategy})")
            expected_counts_data = {}
            all_prob_cols_exist = True
            for label in OUTCOME_LABELS:
                col_name = prob_cols_prefix.format(label=label)
                if col_name in df_results.columns:
                    expected_counts_data[label] = df_results[col_name].sum()
                else:
                    st.warning(f"Probability column '{col_name}' not found for {selected_strategy} expected counts plot.")
                    all_prob_cols_exist = False
                    break

            if all_prob_cols_exist:
                 expected_counts_df = pd.DataFrame(list(expected_counts_data.items()), columns=['Outcome', 'Expected Customers'])
                 all_outcomes_df = pd.DataFrame({'Outcome': OUTCOME_LABELS})
                 expected_counts_df = pd.merge(all_outcomes_df, expected_counts_df, on='Outcome', how='left').fillna(0)

                 plot_single_strategy_expected_counts(expected_counts_df, selected_strategy)
            else:
                 st.info(f"Could not generate expected outcome counts plot for {selected_strategy}.")
            st.subheader(f"Breakdown of {optimal_treatment_label}")
            plot_optimal_treatment_breakdown(df_results, optimal_treatment_col, optimal_treatment_label)
        else:
             st.warning(f"Could not find the required result columns ('{value_col}', '{optimal_treatment_col}') for the selected strategy.")

    st.subheader("Customer Optimal Results Preview")

    if selected_strategy == STRATEGY_COMPARISON:
        display_cols = [
            'customer_id',
            'Maximum Expected Value',
            'Best Treatment Plan for Maximum Value',
            'EV Optimized Accept Probability',
            'Retention-Optimized Accept Prob',
            'Best Treatment Plan for Accepting Renewal',
            'Maximum Expected Value (For Retention)',
            'Historical Expected Value'
        ]
    elif selected_strategy == STRATEGY_EXPECTED_VALUE:
        display_cols = [
            'customer_id',
            'Maximum Expected Value',
            'Best Treatment Plan for Maximum Value',
            'EV Optimized Accept Probability',
            'usage',
            'state'
        ]
    else:
         display_cols = [
             'customer_id',
             'Retention-Optimized Accept Prob',
             'Best Treatment Plan for Accepting Renewal',
             'Maximum Expected Value (For Retention)',
             'usage',
             'state'
         ]
    original_features_in_results = ['usage', 'state', 'cust_tenure', 'before_discount']
    for col in original_features_in_results:
        if col in df_results.columns and col not in display_cols:
            display_cols.append(col)

    display_cols = [col for col in display_cols if col in df_results.columns]

    if display_cols:
        st.dataframe(df_results[display_cols].head())
    else:
        st.warning("Core result columns not found in the results DataFrame.")


elif st.session_state.get(MAIN_DF_SESSION_STATE_KEY) is None and not st.session_state.get('just_cleared', False):
    st.info("Please upload a CSV file in the sidebar to get started.")
