
import pandas as pd
import re
import html
from pathlib import Path

# --------------------------------
# 1. File paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "raw" / "Customer_support_data.csv"

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "Customer_support_data_cleaned.csv"
)

# --------------------------------
# 2. Load dataset
# --------------------------------

print("\nLoading customer support dataset...")

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print("Original rows:", original_rows)
print("Original columns:", original_columns)

# --------------------------------
# 3. Check required column
# --------------------------------

if "Customer Remarks" not in df.columns:
    raise ValueError("Customer Remarks column not found!")

# --------------------------------
# 4. Text cleaning function
# --------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text
    )

    # Remove likely phone numbers
    text = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# --------------------------------
# 5. Clean customer remarks
# --------------------------------

print("\nCleaning Customer Remarks...")

df["Customer Remarks"] = df["Customer Remarks"].apply(clean_text)

# Create NLP input feature
df["customer_text"] = df["Customer Remarks"]

# --------------------------------
# 6. Dataset statistics
# --------------------------------

empty_remarks = (df["customer_text"] == "").sum()

print("\n========== TEXT CLEANING REPORT ==========")

print("Total records:", len(df))
print("Total columns:", len(df.columns))
print("Empty customer remarks:", empty_remarks)
print(
    "Non-empty customer remarks:",
    len(df) - empty_remarks
)

print("\nSample cleaned customer messages:")

for message in df["customer_text"].loc[
    df["customer_text"] != ""
].head(5):
    print("-", message)

# --------------------------------
# 7. Save processed dataset
# --------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\nCleaned dataset saved successfully!")
print("Output:", OUTPUT_FILE)