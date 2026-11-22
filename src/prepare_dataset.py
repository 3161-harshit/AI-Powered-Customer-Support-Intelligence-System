
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ----------------------------------
# 1. Project paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "Customer_support_data_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

# ----------------------------------
# 2. Load dataset
# ----------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# ----------------------------------
# 3. Select features
# ----------------------------------

feature_columns = [
    "customer_text",
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

target_column = "CSAT Score"

# Check required columns
required_columns = feature_columns + [target_column]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Keep only selected features and target
df = df[required_columns].copy()

# ----------------------------------
# 4. Handle missing values
# ----------------------------------

print("\nHandling missing values...")

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)

categorical_columns = [
    "category",
    "Sub-category",
    "channel_name",
    "Tenure Bucket",
    "Agent Shift"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown").astype(str)

# Remove rows with invalid target values
df = df.dropna(subset=[target_column])

# ----------------------------------
# 5. Separate features and target
# ----------------------------------

X = df[feature_columns]
y = df[target_column].astype(int)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts().sort_index())

# ----------------------------------
# 6. Train-test split
# ----------------------------------

print("\nCreating train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------
# 7. Save datasets
# ----------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_data = X_train.copy()
train_data[target_column] = y_train

test_data = X_test.copy()
test_data[target_column] = y_test

train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

# ----------------------------------
# 8. Final report
# ----------------------------------

print("\n========== DATA PREPARATION REPORT ==========")

print("Total records:", len(df))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining dataset saved:", train_path)
print("Testing dataset saved:", test_path)

print("\nDataset preparation completed successfully!")