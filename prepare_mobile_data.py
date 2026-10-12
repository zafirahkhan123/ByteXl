
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).parent


CSV_PATH = (
    BASE_DIR / "data" / "mobile_apps"
    / "Original" / "mobile_apps.csv"
)


TRAIN_DIR = BASE_DIR / "data" / "mobile_apps" / "train"
TEST_DIR = BASE_DIR / "data" / "mobile_apps" / "test"

print("Loading mobile app dataset...")

if not CSV_PATH.exists():
    print("ERROR: Original CSV file nahi mili.")
    print("File path:", CSV_PATH)
    raise SystemExit


df = pd.read_csv(CSV_PATH, low_memory=False)


if "class" not in df.columns:
    print("ERROR: 'class' label column nahi mila.")
    raise SystemExit


df = df.dropna(subset=["class"]).copy()


df = df[df["class"].isin(["B", "S"])].copy()

print("\nOriginal dataset loaded!")
print("Total apps:", len(df))
print("\nClass distribution:")
print(df["class"].value_counts())


train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class"]
)


TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


train_df.to_csv(TRAIN_DIR / "train.csv", index=False)
test_df.to_csv(TEST_DIR / "test.csv", index=False)

print("\n===== SPLIT COMPLETE =====")
print("Training apps:", len(train_df))
print("Testing apps:", len(test_df))

print("\nTraining labels:")
print(train_df["class"].value_counts())

print("\nTesting labels:")
print(test_df["class"].value_counts())

print("\nFiles saved successfully!")
print("Training file:", TRAIN_DIR / "train.csv")
print("Testing file:", TEST_DIR / "test.csv")
