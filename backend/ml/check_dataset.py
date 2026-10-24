
from pathlib import Path
import pandas as pd

csv_path = Path(__file__).parent / "data" / "phishing_urls.csv"

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("Total rows:", len(df))
print("Duplicate rows:", df.duplicated().sum())

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False).sort_index())

print("\nLabel proportions:")
print(
    df["label"].value_counts(normalize=True, dropna=False)
    .sort_index()
    .round(3)
)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFeature summary:")
features = [
    "url_length",
    "num_dots",
    "has_https",
    "has_ip",
    "num_subdirs",
    "num_params",
    "suspicious_words",
    "special_char_count",
    "digits_count",
    "entropy",
]

print(df[features].describe().round(2))

print("\nExamples from each label:")
for label in sorted(df["label"].dropna().unique()):
    print(f"\nLabel {label}:")
    print(df.loc[df["label"] == label, ["url", "label"]].head(3).to_string(index=False))
