
import csv
from pathlib import Path


folder = Path(__file__).parent / "data"
print("\n=== INSTAGRAM DATASET INSPECTION ===\n")


files = list(folder.glob("*.csv"))

if not files:
    print("No CSV files found!")
    print("Check that your CSV files are inside data/social/")
else:
    for file in files:
        print(f"\nFile name: {file.name}")

        with open(file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            columns = next(reader, [])

            print("\nColumn names:")
            for column in columns:
                print("-", column)

            print("\nFirst 3 rows:")
            for i, row in enumerate(reader):
                print(row)
                if i == 2:
                    break

        print("\n" + "=" * 45)

print("\nInspection complete!")
