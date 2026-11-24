import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import MODEL_PATH
from load_data import load_clean_data, split_data
from preprocessing import build_preprocessor


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


def show_feature_importances(pipeline: Pipeline, top_n: int = 20):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    importances = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
    )
    print(f"\n--- Top {top_n} feature importances ---")
    print(importances.round(4))


def train():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Training complete.")
    return pipeline, X_train, X_test, y_train, y_test


def save_model(pipeline: Pipeline, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


def main():
    from evaluate import evaluate

    pipeline, X_train, X_test, y_train, y_test = train()
    evaluate(pipeline, X_train, y_train, X_test, y_test)
    show_feature_importances(pipeline)

    save_model(pipeline)


if __name__ == "__main__":
    main()