
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# --------------------------------
# 1. Paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------
# 2. Load training data only
# --------------------------------

df = pd.read_csv(TRAIN_FILE)

TARGET = "CSAT Score"

FEATURES = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

X = df[FEATURES].fillna("Unknown").astype(str)
y = df[TARGET].astype(int)

print("\nDataset shape:", df.shape)

# --------------------------------
# 3. Validation split
# --------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# 4. Preprocessing
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            FEATURES
        )
    ]
)

# --------------------------------
# 5. Baseline
# --------------------------------

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_val)

print("\n========== MAJORITY BASELINE ==========")
print("Accuracy:", round(
    accuracy_score(y_val, baseline_pred), 4
))
print("Macro F1:", round(
    f1_score(y_val, baseline_pred, average="macro"), 4
))

# --------------------------------
# 6. Compare Logistic Regression
# --------------------------------

results = {}

for name, weight in [
    ("unweighted", None),
    ("balanced", "balanced")
]:

    print(f"\nTraining {name} model...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight=weight,
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    balanced_acc = balanced_accuracy_score(
        y_val, predictions
    )

    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )

    weighted_f1 = f1_score(
        y_val, predictions, average="weighted"
    )

    results[name] = {
        "model": pipeline,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

    print("\nAccuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

# --------------------------------
# 7. Select using validation Macro F1
# --------------------------------

best_name = max(
    results,
    key=lambda name: results[name]["macro_f1"]
)

best_weight = (
    None if best_name == "unweighted" else "balanced"
)

print("\nBest validation model:", best_name)

# --------------------------------
# 8. Retrain selected model on all
#    available training data
# --------------------------------

final_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight=best_weight,
            random_state=42
        )
    )
])

final_model.fit(X, y)

# --------------------------------
# 9. Save model
# --------------------------------

model_path = MODEL_DIR / "csat_operational_model.joblib"

joblib.dump(final_model, model_path)

print("\nSelected model saved:", model_path)
print("\nOperational model experiment completed!")
