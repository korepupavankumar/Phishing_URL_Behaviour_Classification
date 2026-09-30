import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


DATA_PATH = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"

DROP_COLUMNS = [
    "FILENAME",
    "URL",
    "Domain",
    "TLD",
    "Title"
]

TARGET_COLUMN = "label"


def evaluate_model():

    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=DROP_COLUMNS)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = joblib.load(
        "model/phishing_model.pkl"
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\n========== MODEL RESULTS ==========")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC-AUC  :", roc_auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions
    ))


if __name__ == "__main__":
    evaluate_model()