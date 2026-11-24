import pandas as pd
from config import RAW_DATA_PATH, CLEAN_DATA_PATH, TARGET

from sklearn.model_selection import train_test_split

# load raw the data
def load_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)

# load clean data
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA_PATH)

def split_data(df: pd.DataFrame):
    X = df.drop(columns = [TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state= 43, stratify=y
    )
    return X_train, X_test, y_train, y_test

def main():
    load_data()
    load_clean_data()

if __name__ == "__main__":
    main()