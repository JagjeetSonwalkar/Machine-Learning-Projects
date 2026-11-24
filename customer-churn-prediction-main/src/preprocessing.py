import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from load_data import split_data

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler' , StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, NUMERIC_FEATURES),
        ('cat', categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor

def main():
    from load_data import load_clean_data

    df = load_clean_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print(f"X train shpae: {X_train.shape}")
    print(f"X test shape : {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y test shape : {y_test.shape}")

    preprocessor = build_preprocessor()

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  =  preprocessor.transform(X_test)

    print("After preprocessing:", X_train_t.shape, X_test_t.shape)
    print("\nColumns:", list(preprocessor.get_feature_names_out()))
    print("\nMissing after preprocessing:", int(pd.DataFrame(X_train_t).isnull().sum().sum()))
    print("\nScaled train means (should be ~0):", X_train_t[:, :8].mean(axis=0).round(2))
    print("Scaled train stds  (should be ~1):", X_train_t[:, :8].std(axis=0).round(2))



    


if __name__ == "__main__":
    main()