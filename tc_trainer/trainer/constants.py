
VOCAB_DICT = {
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'Dependents': ['Yes', 'No'],
                'DeviceProtection': ['Yes', 'No', 'No internet service'],
                'InternetService': ['DSL', 'Fiber optic', 'No'],
                'MultipleLines': ['Yes', 'No', 'No phone service'],
                'OnlineBackup': ['Yes', 'No', 'No internet service'],
                'OnlineSecurity': ['Yes', 'No', 'No internet service'],
                'PaperlessBilling': ['Yes', 'No'],
                'Partner': ['Yes', 'No'],
                'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check','Mailed check'],
                'PhoneService': ['Yes', 'No'],
                'StreamingMovies': ['Yes', 'No', 'No internet service'],
                'StreamingTV': ['Yes', 'No', 'No internet service'],
                'TechSupport': ['Yes', 'No', 'No internet service'],
                'gender': ['Female', 'Male']
             }

SCHEMA = {
    "customerID": {"na_val": "na", "role": "key", "type": "str"},
    "gender": {"na_val": "na", "role": "input", "type": "cat"},
    "SeniorCitizen": {"na_val": 0.0, "role": "input", "type": "num"},
    "Partner": {"na_val": "na", "role": "input", "type": "cat"},
    "Dependents": {"na_val": "na", "role": "input", "type": "cat"},
    "tenure": {"na_val": 0.0, "role" : "input", "type": "num"},
    "PhoneService": {"na_val": "na", "role": "input", "type": "cat"},
    "MultipleLines": {"na_val": "na", "role": "input", "type": "cat"},
    "InternetService": {"na_val": "na", "role": "input", "type": "cat"},
    "OnlineSecurity": {"na_val": "na", "role": "input", "type": "cat"},
    "OnlineBackup": {"na_val": "na", "role": "input", "type": "cat"},
    "DeviceProtection": {"na_val": "na", "role": "input", "type": "cat"},
    "TechSupport": {"na_val": "na", "role": "input", "type": "cat"},
    "StreamingTV": {"na_val": "na", "role": "input", "type": "cat"},
    "StreamingMovies": {"na_val": "na", "role": "input", "type": "cat"},
    "Contract": {"na_val": "na", "role": "input", "type": "cat"},
    "PaperlessBilling": {"na_val": "na", "role": "input", "type": "cat"},
    "PaymentMethod": {"na_val": "na", "role": "input", "type": "cat"},
    "MonthlyCharges": {"na_val": 0.0, "role": "input", "type": "num"},
    "TotalCharges": {"na_val": 0.0, "role": "input", "type": "num"},
    "Churn": {"na_val": "No", "role": "target", "type": "cat"}
}

    
CSV_COLUMNS = list(SCHEMA.keys())
DEFAULTS = [*map(lambda x: [x['na_val']], SCHEMA.values())]
UNWANTED_COLS = [*map(lambda x: x[0], filter(lambda k: k[1]['role'] == 'key', SCHEMA.items()))]
UNWANTED_COLS += [*map(lambda x: x[0], filter(lambda k: k[1]['role'] == 'drop', SCHEMA.items()))]
LABEL_COLUMN = [*map(lambda x: x[0], filter(lambda k: k[1]['role'] == 'target', SCHEMA.items()))][0]
NUM_COLS = [*map(lambda x: x[0], filter(lambda k: (k[1]['type'] == 'num') and (k[1]['role'] == 'input'), SCHEMA.items()))]
CAT_COLS = [*map(lambda x: x[0], filter(lambda k: (k[1]['type'] == 'cat') and (k[1]['role'] == 'input'), SCHEMA.items()))]
