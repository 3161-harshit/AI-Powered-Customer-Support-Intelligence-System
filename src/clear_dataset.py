
import pandas as pd
import re
import html
from pathlib import Path

# ==========================================
# 1. FILE PATHS
# ==========================================

INPUT_FILE = Path("L:\\AI-Customer-Support-Project\\data\\raw\\Customer_support_data.csv")
OUTPUT_FILE = Path("L:\\AI-Customer-Support-Project\\data\\processed\\Customer_support_data_cleaned.csv")

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading customer support dataset...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE.resolve()}"
    )

try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig",low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_FILE, encoding="cp1252",low_memory=False)

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# ==========================================
# 3. REMOVE DUPLICATE ROWS
# ==========================================

duplicate_count = df.duplicated().sum()

df = df.drop_duplicates().copy()

print("\nDuplicate rows removed:", duplicate_count)

# ==========================================
# 4. IDENTIFY TEXT COLUMNS
# ==========================================

text_keywords = [
    "subject",
    "body",
    "message",
    "description",
    "content",
    "query",
    "issue",
    "ticket_text",
    "customer_text"
]

text_columns = [
    col for col in df.columns
    if any(keyword in col.lower()
        for keyword in text_keywords)
]

print("\nDetected text columns:", text_columns)

# ==========================================
# 5. TEXT CLEANING FUNCTION
# ==========================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

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

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 6. APPLY TEXT CLEANING
# ==========================================

for column in text_columns:

    df[column] = df[column].apply(clean_text)

    print(f"Cleaned column: {column}")

# ==========================================
# 7. CHECK EMPTY TEXT RECORDS
# ==========================================

for column in text_columns:

    empty_count = (df[column] == "").sum()

    print(
        f"Empty records in {column}: {empty_count}"
    )

# ==========================================
# 8. CHECK MISSING VALUES
# ==========================================

print("\nMissing values after cleaning:")

print(df.isnull().sum())

# ==========================================
# 9. SAVE CLEANED DATASET
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n========== CLEANING SUMMARY ==========")

print("Original rows:", len(pd.read_csv(
    INPUT_FILE, encoding="utf-8-sig",
    low_memory=False
)) if False else "See original dataset")

print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Duplicates removed:", duplicate_count)

print("\nCleaned dataset saved at:")
print(OUTPUT_FILE.resolve())

print("\nDataset cleaning completed!")