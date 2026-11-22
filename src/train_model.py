
import pandas as pd
import joblib
import json

from pathlib import Path
from time import perf_counter

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
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

# --------------------------------------
# 1. Project paths
# --------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# 2. Load datasets
# --------------------------------------

print("\nLoading training and testing datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

TARGET = "CSAT Score"

TEXT_FEATURE = "customer_text"

CATEGORICAL_FEATURES = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

FEATURES = [TEXT_FEATURE] + CATEGORICAL_FEATURES

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()

y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

for frame in [X_train, X_test]:
    frame[TEXT_FEATURE] = frame[TEXT_FEATURE].fillna("").astype(str)

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("Unknown").astype(str)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------
# 3. Feature transformation
# --------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            ),
            TEXT_FEATURE
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES
        )
    ]
)

# --------------------------------------
# 4. Build ML pipeline
# --------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42
            )
        )
    ]
)

# --------------------------------------
# 5. Train model
# --------------------------------------

print("\nTraining Logistic Regression model...")

start_time = perf_counter()

model.fit(X_train, y_train)

training_time = perf_counter() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# --------------------------------------
# 6. Make predictions
# --------------------------------------

print("\nPredicting test dataset...")

prediction_start = perf_counter()

y_pred = model.predict(X_test)

prediction_time = perf_counter() - prediction_start

# --------------------------------------
# 7. Evaluate model
# --------------------------------------

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro Precision:   {precision:.4f}")
print(f"Macro Recall:      {recall:.4f}")
print(f"Macro F1-score:    {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")

# --------------------------------------
# 8. Majority-class baseline
# --------------------------------------

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

baseline_macro_f1 = f1_score(
    y_test,
    baseline_pred,
    average="macro",
    zero_division=0
)

print("\n========== BASELINE COMPARISON ==========")

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline Macro F1: {baseline_macro_f1:.4f}")

# --------------------------------------
# 9. Classification report
# --------------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5],
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

# --------------------------------------
# 10. Confusion matrix
# --------------------------------------

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5]
)

print("\n========== CONFUSION MATRIX ==========")
print(matrix)

# --------------------------------------
# 11. Save model and evaluation
# --------------------------------------

model_path = MODEL_DIR / "csat_logistic_regression.joblib"

joblib.dump(model, model_path)

metrics = {
    "model": "Logistic Regression",
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "baseline_accuracy": float(baseline_accuracy),
    "baseline_macro_f1": float(baseline_macro_f1),
    "training_time_seconds": float(training_time),
    "prediction_time_seconds": float(prediction_time)
}

with open(EVALUATION_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

with open(EVALUATION_DIR / "classification_report.txt", "w") as file:
    file.write(report)

print("\nModel saved:", model_path)
print("Metrics saved:", EVALUATION_DIR / "metrics.json")
print("Classification report saved.")

print("\nModel training and evaluation completed!")