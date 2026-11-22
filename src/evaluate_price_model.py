
import pandas as pd
import json
import time

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"
FULL_FILE = BASE_DIR / "data" / "processed" / "Customer_support_data_cleaned.csv"

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
full = pd.read_csv(FULL_FILE)

TARGET = "CSAT Score"

CATEGORICAL_FEATURES = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

NUMERIC_FEATURES = [
    "Item_price",
    "item_price_missing"
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Recreate price features
full["Item_price"] = pd.to_numeric(
    full["Item_price"], errors="coerce"
)

full["item_price_missing"] = (
    full["Item_price"].isna().astype(int)
)

# Match the original train/test split using its saved indices
# The existing train/test files preserve the split order.
# Use the same split indices from the original random split.
from sklearn.model_selection import train_test_split

indices = full.index

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=full[TARGET]
)

X_train = full.loc[train_idx, FEATURES].copy()
X_test = full.loc[test_idx, FEATURES].copy()

y_train = full.loc[train_idx, TARGET]
y_test = full.loc[test_idx, TARGET]

for column in CATEGORICAL_FEATURES:
    X_train[column] = X_train[column].fillna("Unknown")
    X_test[column] = X_test[column].fillna("Unknown")

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ("numeric", numeric_pipeline, NUMERIC_FEATURES)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

print("\n========== PRICE MODEL TRAINING ==========")

start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start

print(f"Training time: {training_time:.4f} seconds")

start = time.perf_counter()
predictions = model.predict(X_test)
prediction_time = time.perf_counter() - start

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
    "macro_precision": precision_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "macro_recall": recall_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "macro_f1": f1_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "weighted_f1": f1_score(
        y_test, predictions, average="weighted", zero_division=0
    ),
    "training_time_seconds": training_time,
    "prediction_time_seconds": prediction_time,
    "average_prediction_time_seconds": prediction_time / len(y_test)
}

print("\n========== TEST RESULTS ==========")

for key, value in metrics.items():
    print(f"{key}: {value:.6f}")

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_test,
    predictions,
    zero_division=0
)

print(report)

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, predictions))

# Save model and evaluation
model_dir = BASE_DIR / "models"
evaluation_dir = BASE_DIR / "evaluation"

model_dir.mkdir(exist_ok=True)
evaluation_dir.mkdir(exist_ok=True)

import joblib

joblib.dump(
    model,
    model_dir / "csat_price_model.joblib"
)

with open(
    evaluation_dir / "price_model_test_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

with open(
    evaluation_dir / "price_model_classification_report.txt",
    "w"
) as f:
    f.write(report)

print("\nModel and evaluation reports saved successfully!")