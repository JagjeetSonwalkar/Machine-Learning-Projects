import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import REPORTS_DIR

LABELS = ["No Churn", "Churn"]


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axes[0], display_labels=LABELS,
        cmap="Blues", values_format="d",
    )
    axes[0].set_title("Counts")

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=axes[1], display_labels=LABELS,
        normalize="true", cmap="Blues", values_format=".2f",
    )
    axes[1].set_title("Normalized (by actual class)")

    fig.suptitle("Confusion Matrix - Test Set")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {l}" for l in LABELS],
        columns=[f"Pred {l}" for l in LABELS],
    )
    print("\n--- Confusion matrix (test) ---")
    print(cm_df)

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


def evaluate(pipeline, X_train, y_train, X_test, y_test):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    metrics = {
        "train": compute_metrics(y_train, train_pred),
        "test": compute_metrics(y_test, test_pred),
    }

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics["test"]["roc_auc"] = float(roc_auc_score(y_test, test_proba))

    print("\n--- Evaluation ---")