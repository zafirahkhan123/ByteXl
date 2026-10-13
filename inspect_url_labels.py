
import pandas as pd

file_path = "data/url/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path, low_memory=False)

print("Dataset shape:", df.shape)

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)

print("\nSample URLs by label:")
for label in sorted(df["label"].unique()):
    print(f"\nLabel {label}:")
    print(df.loc[df["label"] == label, "URL"].head(5).to_list())

