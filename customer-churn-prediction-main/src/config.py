from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
CLEAN_DATA_PATH = BASE_DIR / 'data' / 'cleaned_data' / 'Customer_Churn.csv'
MODEL_PATH = BASE_DIR / 'models' / 'model.pkl'
REPORTS_DIR = BASE_DIR / "reports"

NUMERIC_FEATURES = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

CATEGORICAL_FEATURES = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

TARGET ='Churn'