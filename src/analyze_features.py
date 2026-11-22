
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "Customer_support_data_cleaned.csv"
)

# Load cleaned dataset
print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# 1. Target variable analysis
print("\n========== CSAT SCORE DISTRIBUTION ==========")

if "CSAT Score" in df.columns:
    print(df["CSAT Score"].value_counts(dropna=False).sort_index())

    print("\nCSAT percentages:")
    print(
        (df["CSAT Score"].value_counts(normalize=True, dropna=False) * 100)
        .round(2)
    )
else:
    print("CSAT Score column not found.")

# 2. Category analysis
print("\n========== CATEGORY DISTRIBUTION ==========")

for column in ["category", "Sub-category", "channel_name"]:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).head(15))

# 3. Missing value analysis
print("\n========== MISSING VALUE ANALYSIS ==========")

missing = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame({
    "Missing Values": missing,
    "Missing Percentage": missing_percentage
})

print(missing_report.sort_values(
    by="Missing Percentage",
    ascending=False
))

# 4. Customer text analysis
print("\n========== CUSTOMER TEXT ANALYSIS ==========")

if "customer_text" in df.columns:

    text_lengths = df["customer_text"].fillna("").astype(str).str.len()

    print("Average text length:", round(text_lengths.mean(), 2))
    print("Maximum text length:", text_lengths.max())
    print("Minimum text length:", text_lengths.min())

    print(
        "Empty text records:",
        (text_lengths == 0).sum()
    )

# 5. Duplicate analysis
print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate rows:", df.duplicated().sum())

print("\nFeature analysis completed!")
