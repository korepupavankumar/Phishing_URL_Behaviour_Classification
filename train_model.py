import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

DATA_PATH = "data/raw/PhiUSIIL_Phishing_URL_Dataset.csv"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "url_phishing_model.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "url_feature_columns.pkl"
)


# ============================================================
# URL-BASED FEATURES
# ============================================================

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]


# ============================================================
# START
# ============================================================

print("=" * 60)
print("PHISHING URL BEHAVIOUR CLASSIFICATION")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = URL_FEATURES + ["label"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")

    for column in missing_columns:
        print("-", column)

    raise ValueError(
        "Required columns are missing from the dataset."
    )


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X = df[URL_FEATURES]

y = df["label"]


print("\nNumber of URL features:", len(URL_FEATURES))

print("X shape:", X.shape)

print("y shape:", y.shape)


# ============================================================
# LABEL INFORMATION
# ============================================================

print("\nDataset label mapping:")
print("0 = Phishing")
print("1 = Legitimate")

print("\nClass distribution:")

print(
    y.value_counts().sort_index()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# CREATE LOGISTIC REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)


# ============================================================
# PHISHING PROBABILITY
# ============================================================
#
# PhiUSIIL label mapping:
#
# 0 = Phishing
# 1 = Legitimate
#
# Therefore:
#
# probabilities[:, 0] = Phishing probability
# probabilities[:, 1] = Legitimate probability
# ============================================================

phishing_probability = probabilities[:, 0]


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# Class 0 = Phishing
precision = precision_score(
    y_test,
    y_pred,
    pos_label=0
)


# Class 0 = Phishing
recall = recall_score(
    y_test,
    y_pred,
    pos_label=0
)


# Class 0 = Phishing
f1 = f1_score(
    y_test,
    y_pred,
    pos_label=0
)


# Convert:
# 0 = Phishing -> 1
# 1 = Legitimate -> 0
#
# This allows ROC-AUC to treat Phishing
# as the positive class.

phishing_actual = (
    y_test == 0
).astype(int)


roc_auc = roc_auc_score(
    phishing_actual,
    phishing_probability
)


# ============================================================
# DISPLAY EVALUATION RESULTS
# ============================================================

print("\n")
print("=" * 40)
print("MODEL EVALUATION RESULTS")
print("=" * 40)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(
    "(Rows = Actual, Columns = Predicted)"
)

print(
    "(0 = Phishing, 1 = Legitimate)"
)

print()

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Phishing",
            "Legitimate"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


# ============================================================
# SAVE FEATURE COLUMN ORDER
# ============================================================

joblib.dump(
    URL_FEATURES,
    FEATURE_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 40)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 40)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nFeature columns:"
)

print(
    FEATURE_PATH
)

print(
    "\nTraining process completed successfully."
)