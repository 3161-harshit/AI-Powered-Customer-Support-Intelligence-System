
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "processed" / "Customer_support_data_cleaned.csv"

df = pd.read_csv(FILE)

TARGET = "CSAT Score"

operational_features = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

df["Item_price"] = pd.to_numeric(
    df["Item_price"], errors="coerce"
)

df["item_price_missing"] = (
    df["Item_price"].isna().astype(int)
)

# Keep only records with all target values available
df = df.dropna(subset=[TARGET]).copy()

X_base = df[operational_features].fillna("Unknown")

X_price = df[
    operational_features + ["Item_price"]
].copy()

X_price_indicator = df[
    operational_features + [
        "Item_price",
        "item_price_missing"
    ]
].copy()

y = df[TARGET]

# Same validation split for every experiment
indices = df.index

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

def evaluate_model(X, name, numeric_features):

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers = [
        ("categorical", categorical_pipeline, categorical_features)
    ]

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    preprocessing = ColumnTransformer(transformers)

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X.loc[train_idx], y.loc[train_idx])

    predictions = model.predict(X.loc[val_idx])

    score = f1_score(
        y.loc[val_idx],
        predictions,
        average="macro"
    )

    print(f"{name}: Macro F1 = {score:.4f}")

    return score


print("\n========== ITEM PRICE FEATURE EXPERIMENT ==========")

results = {}

results["Operational only"] = evaluate_model(
    X_base,
    "Operational only",
    []
)

results["Operational + price"] = evaluate_model(
    X_price,
    "Operational + price",
    ["Item_price"]
)

results["Operational + price + missing indicator"] = evaluate_model(
    X_price_indicator,
    "Operational + price + missing indicator",
    ["Item_price", "item_price_missing"]
)

print("\n========== FINAL COMPARISON ==========")

for name, score in results.items():
    print(f"{name}: {score:.4f}")

best = max(results, key=results.get)

print("\nBest validation configuration:", best)
print("\nExperiment completed!")