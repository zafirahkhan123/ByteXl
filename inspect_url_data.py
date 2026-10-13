
import pandas as pd
from pathlib import Path

file_path = Path("data/url/PhiUSIIL_Phishing_URL_Dataset.csv")

if not file_path.exists():
    print("CSV file nahi mili!")
    print("Expected location:", file_path.resolve())
else:
    df = pd.read_csv(file_path, nrows=5, low_memory=False)

    print("\nDataset successfully opened!")
    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn data types:")
    print(df.dtypes)
