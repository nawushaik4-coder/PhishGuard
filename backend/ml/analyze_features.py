
from pathlib import Path
import pandas as pd

path = Path(__file__).parent / "data" / "phishing_urls.csv"
df = pd.read_csv(path)

features = [
    "url_length", "num_dots", "has_https", "has_ip",
    "num_subdirs", "num_params", "suspicious_words",
    "special_char_count", "digits_count", "entropy",
]

for label, name in [(0, "Legitimate"), (1, "Phishing")]:
    subset = df[df["label"] == label]

    print(f"\n{name} (label {label})")
    print("Rows:", len(subset))
    print("Unique URLs:", subset["url"].nunique())
    print("Unique feature patterns:", len(subset[features].drop_duplicates()))

    print("\nFeature means:")
    print(subset[features].mean().round(3))

print("\nFeature pattern counts by label:")
patterns = df.groupby(features, dropna=False)["label"].agg(
    rows="size",
    distinct_labels="nunique"
)
print("Patterns shared across labels:", (patterns["distinct_labels"] > 1).sum())
