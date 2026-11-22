
import pandas as pd
import joblib
import json

from pathlib import Path
from time import perf_counter

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

# --------------------------------
# 1. Project paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"

MODEL_FILE = BASE_DIR / "models" / "csat_operational_model.joblib"

EVALUATION_DIR = BASE_DIR / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. Load test dataset
# --------------------------------

print("\nLoading untouched test dataset...")

df = pd.read_csv(TEST_FILE)

TARGET = "CSAT Score"

FEATURES = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

X_test = df[FEATURES].fillna("Unknown").astype(str)
y_test = df[TARGET].astype(int)

print("Test records:", len(df))

# --------------------------------
# 3. Load trained model
# --------------------------------

print("\nLoading operational model...")

model = joblib.load(MODEL_FILE)

# --------------------------------
# 4. Make predictions
# --------------------------------

print("\nGenerating predictions...")

start_time = perf_counter()

y_pred = model.predict(X_test)

prediction_time = perf_counter() - start_time

# --------------------------------
# 5. Evaluate model
# --------------------------------

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "macro_precision": precision_score(
        y_test, y_pred, average="macro", zero_division=0
    ),
    "macro_recall": recall_score(
        y_test, y_pred, average="macro", zero_division=0
    ),
    "macro_f1": f1_score(
        y_test, y_pred, average="macro", zero_division=0
    ),
    "weighted_f1": f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
}

print("\n========== FINAL TEST PERFORMANCE ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print(f"Prediction time: {prediction_time:.4f} seconds")
print(
    f"Average prediction time per record: "
    f"{prediction_time / len(X_test):.6f} seconds"
)

# --------------------------------
# 6. Majority baseline comparison
# --------------------------------

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(
    pd.read_csv(BASE_DIR / "data" / "processed" / "train.csv")[FEATURES],
    pd.read_csv(BASE_DIR / "data" / "processed" / "train.csv")[TARGET]
)

baseline_pred = baseline.predict(X_test)

print("\n========== BASELINE COMPARISON ==========")

print(
    "Baseline Accuracy:",
    round(accuracy_score(y_test, baseline_pred), 4)
)

print(
    "Baseline Macro F1:",
    round(
        f1_score(
            y_test, baseline_pred,
            average="macro",
            zero_division=0
        ),
        4
    )
)

# --------------------------------
# 7. Classification report
# --------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5],
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

# --------------------------------
# 8. Confusion matrix
# --------------------------------

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4, 5]
)

print("\n========== CONFUSION MATRIX ==========")
print(matrix)

# --------------------------------
# 9. Save evaluation
# --------------------------------

output_file = EVALUATION_DIR / "operational_test_metrics.json"

with open(output_file, "w") as file:
    json.dump(
        {
            **{key: float(value) for key, value in metrics.items()},
            "prediction_time_seconds": prediction_time,
            "test_records": len(X_test)
        },
        file,
        indent=4
    )

with open(
    EVALUATION_DIR / "operational_classification_report.txt",
    "w"
) as file:
    file.write(report)

print("\nEvaluation saved:", output_file)
print("\nFinal test evaluation completed!")