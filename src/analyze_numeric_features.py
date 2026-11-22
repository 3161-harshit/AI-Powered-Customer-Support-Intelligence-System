
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "processed" / "Customer_support_data_cleaned.csv"

df = pd.read_csv(FILE)

TARGET = "CSAT Score"

NUMERIC_FEATURES = [
    "Item_price",
    "connected_handling_time"
]

print("\n========== NUMERIC FEATURE ANALYSIS ==========")

for column in NUMERIC_FEATURES:

    if column not in df.columns:
        print(f"\n{column}: Column not found")
        continue

    df[column] = pd.to_numeric(df[column], errors="coerce")

    print(f"\n========== {column.upper()} ==========")

    print("Available records:", df[column].notna().sum())
    print("Missing records:", df[column].isna().sum())

    print("\nStatistical summary:")
    print(df[column].describe())

    print("\nAverage value by CSAT score:")

    print(
        df.groupby(TARGET)[column]
        .agg(["count", "mean", "median"])
        .round(2)
    )

    print("\nCorrelation with CSAT:")
    print(df[[column, TARGET]].corr().round(4).iloc[0, 1])

print("\nNumeric feature analysis completed!")