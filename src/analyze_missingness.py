
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "processed" / "Customer_support_data_cleaned.csv"

df = pd.read_csv(FILE)

FEATURES = [
    "Item_price",
    "connected_handling_time"
]

print("\n========== MISSING VALUE ANALYSIS ==========")

for column in FEATURES:

    print(f"\n========== {column.upper()} ==========")

    df[column] = pd.to_numeric(df[column], errors="coerce")

    df[f"{column}_missing"] = df[column].isna()

    print("\nMissing vs available:")
    print(df[f"{column}_missing"].value_counts())

    print("\nMissing percentage by CSAT score:")

    result = (
        df.groupby("CSAT Score")[f"{column}_missing"]
        .mean() * 100
    )

    print(result.round(2))

    print("\nMissing percentage by category:")

    category_result = (
        df.groupby("category")[f"{column}_missing"]
        .mean() * 100
    )

    print(category_result.round(2).sort_values(ascending=False))

print("\nMissingness analysis completed!")