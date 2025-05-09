
import streamlit as st

from config import (
    MODEL_PATH, DISCOUNT_RANGE, OUTCOME_VALUES, RETENTION_CLASS_LABEL,
    POSSIBLE_TREATMENTS,
    STRATEGY_OPTIONS, STRATEGY_RETENTION, STRATEGY_EXPECTED_VALUE,
    DEFAULT_CUSTOMER_LIMIT, RANDOM_STATE_SAMPLING
)
# Import preprocessing functions
from data_handler import load_csv, preprocess_dataframe
# Import the consistent key definition from database.py
from database import MAIN_DF_SESSION_STATE_KEY, load_dataframe_from_session_state
# Import calculation functions, including the new per-customer evaluator
from inference_handler import evaluate_scenarios_per_customer
# Import model loading and prediction functions
from model_handler import load_ml_model, predict_probabilities
# Import visualization functions, potentially including a new one for treatment breakdown
from visualization_handler import (
    plot_distribution,
    # This one might be less relevant now, replaced by colored version
    plot_average_value_by_category,
    plot_optimal_treatment_breakdown,
    plot_value_vs_original_feature_colored  # Add the new colored scatter plot
)

# We will now use MAIN_DF_SESSION_STATE_KEY directly in app.py for consistency
# from database import save_dataframe_to_session_state, clear_session_state_data # These functions might become unused

st.set_page_config(
    page_title="Energy Retail Customer Renewal Strategy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Energy Retail Customer Renewal Strategy Dashboard")

# --- Session State Initialization ---
if 'uploaded_data_processed' not in st.session_state:
    st.session_state['uploaded_data_processed'] = False
# Store results from per-customer evaluation
if 'df_per_customer_results' not in st.session_state:
    st.session_state['df_per_customer_results'] = None
if 'calculate_button_clicked' not in st.session_state:
    st.session_state['calculate_button_clicked'] = False
if 'selected_strategy' not in st.session_state:
    st.session_state['selected_strategy'] = STRATEGY_EXPECTED_VALUE
if 'processed_customer_count_input' not in st.session_state:
     st.session_state['processed_customer_count_input'] = DEFAULT_CUSTOMER_LIMIT
if 'actual_processed_count' not in st.session_state:
    st.session_state['actual_processed_count'] = 0




# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])

    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            # Store the full original DataFrame directly using the consistent key
            st.session_state[MAIN_DF_SESSION_STATE_KEY] = df # Use the key from database.py config

            st.session_state['uploaded_data_processed'] = True
            st.session_state['df_per_customer_results'] = None # Clear previous results
            st.session_state['calculate_button_clicked'] = False # Reset
            st.session_state['actual_processed_count'] = 0 # Reset
            st.success(f"CSV uploaded successfully. Contains {len(df)} rows.")
            # Set initial limit based on uploaded data size
            st.session_state['processed_customer_count_input'] = min(len(df), DEFAULT_CUSTOMER_LIMIT)



    st.header("Choose Strategy")
    selected_strategy = st.radio(
        "Select Optimization Goal:",
        STRATEGY_OPTIONS,
        key='strategy_radio'
    )
    st.session_state['selected_strategy'] = selected_strategy


    st.header("Simulation Settings")
    # **Load using the function which now uses the correct key**
    df_original = load_dataframe_from_session_state() # This function uses MAIN_DF_SESSION_STATE_KEY

    user_discount = None # Initialize discount

    if df_original is not None:
       total_customers = len(df_original)
       st.info(f"Total customers in uploaded data: {total_customers}")

       customer_limit_input = st.number_input(
           "Limit number of customers to process:",
           min_value=1,
           max_value=total_customers,
           value=st.session_state.get('processed_customer_count_input', DEFAULT_CUSTOMER_LIMIT), # Use get with default
           step=1,
           help="Process a subset of customers to manage calculation time. Uses random sampling.",
           key='customer_limit_input'
       )
       st.session_state['processed_customer_count_input'] = customer_limit_input


       user_discount = st.slider(
           f"Set Discount for '{POSSIBLE_TREATMENTS[2]}':", # Assuming "Lower Discount" is the 3rd option
           min_value=DISCOUNT_RANGE[0],
           max_value=DISCOUNT_RANGE[1],
           value=st.session_state.get('discount_slider', DISCOUNT_RANGE[0]), # Use get with default
           step=0.01,
           format="%.2f",
           key='discount_slider',
           help="This discount is applied when evaluating the 'Lower Discount' treatment scenario."
       )

    else:
       st.info("Upload data first to set simulation settings.")


    st.header("Actions")
    # Check if the main data key has a dataframe
    if st.session_state.get(MAIN_DF_SESSION_STATE_KEY) is not None and user_discount is not None:
        if st.button("Run Simulation & Calculate Results"):
            st.session_state['calculate_button_clicked'] = True # Trigger calculation
            print("calculate_button_clicked -- True   ",  st.session_state['calculate_button_clicked'])

        if st.button("Clear Uploaded Data"):
            keys_to_clear = [
                MAIN_DF_SESSION_STATE_KEY,
                'uploaded_data_processed',
                'df_per_customer_results',
                'calculate_button_clicked',
                'actual_processed_count',
                'processed_customer_count_input',
                'selected_strategy',
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
    elif st.session_state.get(MAIN_DF_SESSION_STATE_KEY) is not None:
         st.warning("Cannot calculate without setting simulation inputs.")
    else:
         st.info("Upload data to enable simulations.")




df_original = load_dataframe_from_session_state()


if st.session_state.get('calculate_button_clicked', False) and df_original is not None and user_discount is not None:
    print("Start Calculation---")
    st.subheader("Running Simulation...")

    customer_limit = st.session_state['processed_customer_count_input']
    actual_limit = min(len(df_original), customer_limit)

    df_sample = df_original.sample(n=actual_limit, random_state=RANDOM_STATE_SAMPLING).reset_index(drop=True)
    st.session_state['actual_processed_count'] = len(df_sample) # Store actual count processed
    st.info(f"Processing {len(df_sample)} customers...")


    with st.spinner("Loading ML model..."):
        print('Loading the ML Model')
        model_pipeline = load_ml_model(MODEL_PATH)
        if model_pipeline is None:
            print('Loading Failed')
            st.error("Model loading failed. Cannot proceed.")
            st.session_state['calculate_button_clicked'] = False
            st.stop()
        st.success("Model loaded.")

    with st.spinner(f"Evaluating {len(POSSIBLE_TREATMENTS)} scenarios for each of the {len(df_sample)} customers..."):

        print('Calling the evaluation function....')
        df_per_customer_results = evaluate_scenarios_per_customer(
             df_sample,
             POSSIBLE_TREATMENTS,
             user_discount,
             model_pipeline, # Pass the loaded pipeline
             OUTCOME_VALUES,
             RETENTION_CLASS_LABEL
         )

        st.session_state['df_per_customer_results'] = df_per_customer_results # Store results
        st.success("Simulation complete.")

    st.session_state['calculate_button_clicked'] = False # Reset button state after calculation
    # Rerun after successful calculation to display results
    #st.rerun()


# --- Display Results and Visualizations ---
# Check if results are available and not just cleared
if st.session_state['df_per_customer_results'] is not None and not st.session_state['df_per_customer_results'].empty and not st.session_state.get('just_cleared', False):

    processed_count = st.session_state['actual_processed_count']
    st.subheader(f"Simulation Results for {processed_count} Customers")
    st.info(f"Analysis based on optimizing for: **{st.session_state['selected_strategy']}**")


    # Determine which columns and labels to use based on strategy
    if st.session_state['selected_strategy'] == STRATEGY_RETENTION:
        value_col = 'max_prob_accepts'
        value_label = f'Maximum Predicted Probability of {RETENTION_CLASS_LABEL}'
        optimal_treatment_col = 'optimal_treatment_prob'
        optimal_treatment_label = 'Optimal Treatment (Maximize Retention)'
    else: # Maximize Expected Value
        value_col = 'max_expected_value'
        value_label = 'Maximum Expected Value'
        optimal_treatment_col = 'optimal_treatment_ev'
        optimal_treatment_label = 'Optimal Treatment (Maximize EV)'

    df_results = st.session_state['df_per_customer_results']

    st.subheader(f"Customer Optimal Results Preview ({value_label})")
    # Define display columns including some original features now returned by evaluation function
    display_cols = ['customer_id', 'usage', 'state', 'cust_tenure', value_col, optimal_treatment_col]
    display_cols = [col for col in display_cols if col in df_results.columns] # Ensure columns exist
    if display_cols:
        st.dataframe(df_results[display_cols].head())
    else:
        st.warning("Core result columns not found in the results DataFrame.")


    if value_col in df_results.columns and optimal_treatment_col in df_results.columns:
        st.subheader(f"Distribution of {value_label} (Per Customer)")
        plot_distribution(df_results, value_col, value_label)

        st.subheader(f"Breakdown of {optimal_treatment_label}")
        plot_optimal_treatment_breakdown(df_results, optimal_treatment_col, optimal_treatment_label)


        # Optional: Plot Avg Value/Prob by Optimal Treatment
        st.subheader(f"Average {value_label} by {optimal_treatment_label}")
        plot_average_value_by_category(df_results, optimal_treatment_col, value_col, optimal_treatment_label, value_label)

        # Plot Value vs Original Feature colored by Optimal Treatment
        # Use the original features that were included in df_per_customer_results
        original_features_to_plot = ['usage', 'cust_tenure', 'before_discount', 'state'] # Example original features included in results
        for original_feature_col in original_features_to_plot:
             original_feature_label = original_feature_col.replace('_', ' ').title() # Simple label formatting
             if original_feature_col in df_results.columns: # Check if the feature was included in results
                 st.subheader(f"{value_label} vs. {original_feature_label} (Colored by {optimal_treatment_label})")
                 plot_value_vs_original_feature_colored(
                      df_results,
                      original_feature_col,
                      value_col,
                      optimal_treatment_col,
                      original_feature_label,
                      value_label,
                      optimal_treatment_label
                  )


    else:
         st.warning(f"Could not find the required result columns ('{value_col}', '{optimal_treatment_col}') for the selected strategy.")


# Initial message when no data is loaded and not just cleared
elif st.session_state.get(MAIN_DF_SESSION_STATE_KEY) is None and not st.session_state.get('just_cleared', False):
    st.info("Please upload a CSV file in the sidebar to get started.")

