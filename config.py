
MODEL_PATH = 'saved_model/xgboost.pkl'

CATEGORICAL_FEATURES_FOR_PREPROCESSING = [
    'state',
    'communication_preference',
    'age',
    'before_channel',
    'treatment_given',
    'renewal_month'
]

NUMERICAL_FEATURES_FOR_PREPROCESSING = [
    'usage',
    'cust_tenure',
    'years_on_disc',
    'before_discount',
    'discount_offered',
    'discount_change',
    'green',
    'dual_fuel_customer',
    'direct_debit_flag'
]

ALL_FEATURES_BEFORE_TRANSFORMER = NUMERICAL_FEATURES_FOR_PREPROCESSING + CATEGORICAL_FEATURES_FOR_PREPROCESSING


COLUMNS_TO_DROP_RAW = ['customer_id', 'renewal_date']


TREATMENT_INPUT_COL = 'treatment_given'
DISCOUNT_INPUT_COL = 'discount_offered'


POSSIBLE_TREATMENTS = ["Same Contract", "Remove Discount", "Lower Discount"]
DISCOUNT_RANGE = (0.0, 0.5)

STRATEGY_RETENTION = "Maximize Customer Retention"
STRATEGY_EXPECTED_VALUE = "Maximize Expected Value"
STRATEGY_COMPARISON = "Compare Strategies"
STRATEGY_OPTIONS = [STRATEGY_EXPECTED_VALUE, STRATEGY_RETENTION, STRATEGY_COMPARISON]



OUTCOME_LABELS = ['Accepted', 'Call_Back', 'Churn']

RETENTION_CLASS_LABEL = 'Accepted'

OUTCOME_VALUES = {
    'Accepts': 200.0,
    'Call Back': 50.0,
    'Churn': -150.0
}


DEFAULT_CUSTOMER_LIMIT = 10000
MAX_CUSTOMER_LIMIT = 1000000
RANDOM_STATE_SAMPLING = 42

