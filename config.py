# config.py

# --- File Paths ---
MODEL_PATH = 'saved_model/random_forest_exp1.joblib' # Path to your trained ML model pipeline

# --- Database Configuration (Placeholder) ---
# Not directly used in the current session state simulation but good to keep structure
DB_CONFIG = {
    "host": "localhost",
    "database": "your_db_name",
    "user": "your_db_user",
    "password": "your_db_password"
}
DB_TABLE_NAME = "uploaded_customer_data"

# --- Data Configuration ---
# These lists define the columns *after* initial feature engineering (like binary conversion,
# discount_change, renewal_month extraction) but *before* the ColumnTransformer.
# These should match the lists passed to create_preprocessing_pipeline in your training code.

# Categorical features that go into the OneHotEncoder
CATEGORICAL_FEATURES_FOR_PREPROCESSING = [
    'state',
    'communication_preference',
    'age',              # Your training code treats age as categorical here
    'before_channel',
    'treatment_given',  # This is the dynamic input feature
    'renewal_month'     # Engineered feature
]

# Numerical features that go into the StandardScaler (includes engineered and converted binary)
NUMERICAL_FEATURES_FOR_PREPROCESSING = [
    'usage',
    'cust_tenure',
    'years_on_disc',
    'before_discount',
    'discount_offered', # This is the dynamic input feature
    'discount_change',  # Engineered feature
    'green',            # Converted binary
    'dual_fuel_customer', # Converted binary
    'direct_debit_flag' # Converted binary
]

# List of all feature columns expected *before* the ColumnTransformer,
# after initial engineering. Used for selecting/ordering columns consistently.
ALL_FEATURES_BEFORE_TRANSFORMER = NUMERICAL_FEATURES_FOR_PREPROCESSING + CATEGORICAL_FEATURES_FOR_PREPROCESSING


# Original columns to drop after loading
COLUMNS_TO_DROP_RAW = ['customer_id', 'renewal_date']


# User Input Column Names (These are original columns that get set dynamically)
TREATMENT_INPUT_COL = 'treatment_given'
DISCOUNT_INPUT_COL = 'discount_offered'


# Possible values for treatments evaluated per customer - MUST match values your model can handle after preprocessing
POSSIBLE_TREATMENTS = ["Same Contract", "Remove Discount", "Lower Discount"]
DISCOUNT_RANGE = (0.0, 0.5) # Range for the user discount slider

# --- Strategy Configuration ---
STRATEGY_RETENTION = "Maximize Customer Retention"
STRATEGY_EXPECTED_VALUE = "Maximize Expected Value"
STRATEGY_OPTIONS = [STRATEGY_RETENTION, STRATEGY_EXPECTED_VALUE]

# The exact labels your ML model uses for the outcomes (order might matter for predict_proba array)
# CRITICAL: Update these to match your model.classes_
OUTCOME_LABELS = ['Accepted', 'Call_Back', 'Churn']

# The label for the positive retention outcome (must be in OUTCOME_LABELS)
RETENTION_CLASS_LABEL = 'Accepted' # <-- **CRITICAL: Update to match the 'Accepts' label from your model**


# --- Expected Value Configuration ---
# Define the value associated with each outcome.
# CRITICAL: Update these values based on your business understanding of each outcome.
# Example values - replace with your actual profit/cost figures
OUTCOME_VALUES = {
    'Accepts': 200.0,     # Example: High positive value for acceptance
    'Call Back': 50.0,    # Example: Small positive value for a call back/engagement
    'Churn': -150.0       # Example: Negative value for churn (cost of acquisition, lost revenue etc.)
}
# Ensure all labels in OUTCOME_LABELS are also keys in OUTCOME_VALUES if you want them included in EV

# --- Performance Configuration ---
DEFAULT_CUSTOMER_LIMIT = 10000 # Default number of customers to process
MAX_CUSTOMER_LIMIT = 1000000 # Max value for the limit input (adjust based on expected dataset size)
RANDOM_STATE_SAMPLING = 42 # Seed for reproducible sampling

# --- Visualization Configuration ---
# Add any plot specific configs here
