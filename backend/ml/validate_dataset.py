
from pathlib import Path
import pandas as pd

path = Path(__file__).parent / "data" / "phishing_urls.csv"
df = pd.read_csv(path)

features = [
    "url_length", "num_dots", "has_https", "has_ip",
    "num_subdirs", "num_params", "suspicious_words",
    "special_char_count", "digits_count", "entropy",
]

print("Original rows:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())
print("Duplicate URLs:", df["url"].duplicated().sum())

print("\nLabels by class:")
print(df["label"].value_counts().sort_index())

print("\nUnique feature patterns:", len(df[features].drop_duplicates()))

print("\nLabels for duplicate feature patterns:")
pattern_labels = df.groupby(features, dropna=False)["label"].nunique()
print("Patterns associated with multiple labels:", (pattern_labels > 1).sum())

print("\nURLs associated with conflicting labels:")
url_labels = df.groupby("url")["label"].nunique()
print("Conflicting URLs:", (url_labels > 1).sum())

print("\nExample legitimate URLs:")
print(df.loc[df["label"] == 0, "url"].head(10).to_string(index=False))

print("\nExample phishing URLs:")
print(df.loc[df["label"] == 1, "url"].head(10).to_string(index=False))
