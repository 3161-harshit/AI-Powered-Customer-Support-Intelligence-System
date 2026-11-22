
import pandas as pd

from pathlib import Path
from sklearn.metrics import normalized_mutual_info_score

# --------------------------------
# 1. Load training dataset only
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "processed" / "train.csv"

df = pd.read_csv(FILE)

TARGET = "CSAT Score"

FEATURES = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

print("\n========== TARGET RELATIONSHIP ANALYSIS ==========")

print("Training records:", len(df))

# --------------------------------
# 2. Feature relationship analysis
# --------------------------------

results = []

for column in FEATURES:

    feature = df[column].fillna("Unknown").astype(str)
    target = df[TARGET].astype(str)

    nmi = normalized_mutual_info_score(
        feature,
        target
    )

    results.append({
        "Feature": column,
        "Unique Values": feature.nunique(),
        "Normalized Mutual Information": round(nmi, 4)
    })

report = pd.DataFrame(results)

report = report.sort_values(
    by="Normalized Mutual Information",
    ascending=False
)

print("\n========== FEATURE ASSOCIATION ==========")
print(report.to_string(index=False))

# --------------------------------
# 3. CSAT distribution by feature
# --------------------------------

for column in FEATURES:

    print(f"\n========== CSAT BY {column.upper()} ==========")

    distribution = pd.crosstab(
        df[column].fillna("Unknown"),
        df[TARGET],
        normalize="index"
    ) * 100

    print(distribution.round(2).head(15))

# --------------------------------
# 4. Feature cardinality
# --------------------------------

print("\n========== FEATURE CARDINALITY ==========")

for column in FEATURES:

    print(
        f"{column}: {df[column].nunique()} unique values"
    )

print("\nTarget relationship analysis completed!")