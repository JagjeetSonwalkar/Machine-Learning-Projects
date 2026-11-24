import joblib
import pandas as pd

from config import MODEL_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

YES_NO = ["Yes", "No"]
NO_INTERNET = ["Yes", "No", "No internet service"]

# valid choices for each categorical feature (same values as the Telco dataset)
OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": NO_INTERNET,
    "OnlineBackup": NO_INTERNET,
    "DeviceProtection": NO_INTERNET,
    "TechSupport": NO_INTERNET,
    "StreamingTV": NO_INTERNET,
    "StreamingMovies": NO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run train.py first.")
    return joblib.load(path)


def ask_choice(name: str, options: list):
    print(f"\n{name}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        answer = input("Choose a number: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        print(f"Please enter a number from 1 to {len(options)}.")


def ask_number(name: str, cast=float, min_value=0, allow_blank=False):
    while True:
        answer = input(f"\n{name}: ").strip()
        if allow_blank and answer == "":
            return " "  # cleaned to 0.0 later, same as training
        try:
            value = cast(answer)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        return value


def get_input() -> pd.DataFrame:
    print("Enter the customer's details.")
    row = {
        "SeniorCitizen": ask_choice("SeniorCitizen (0 = No, 1 = Yes)", [0, 1]),
        "tenure": ask_number("tenure (months with the company)", cast=int),
        "MonthlyCharges": ask_number("MonthlyCharges"),
        "TotalCharges": ask_number("TotalCharges (leave blank if new customer)", allow_blank=True),
    }
    for feature in CATEGORICAL_FEATURES:
        row[feature] = ask_choice(feature, OPTIONS[feature])
    return pd.DataFrame([row])


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[FEATURES].copy()

    # same rule as training: blank TotalCharges -> 0.0, then float
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", "0.0", regex=True), errors="coerce"
    )

    # check categorical values are ones the model has seen
    for col, valid in OPTIONS.items():
        bad = set(df[col]) - set(valid)
        if bad:
            raise ValueError(f"Invalid value(s) for {col}: {bad}. Expected one of {valid}")

    return df


def predict(model, X: pd.DataFrame, threshold: float = 0.5):
    proba = float(model.predict_proba(X)[0, 1])
    label = "Churn" if proba >= threshold else "No Churn"
    return proba, label


def main():
    model = load_model()

    while True:
        df = get_input()
        X = clean_and_validate(df)
        proba, label = predict(model, X)

        print("\n--- Result ---")
        print(f"Churn probability: {proba:.1%}")
        print(f"Prediction       : {label}")

        again = input("\nPredict another customer? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()