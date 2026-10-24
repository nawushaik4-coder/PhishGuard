
from pathlib import Path
import pandas as pd

csv_path = Path(__file__).parent / "data" / "phishing_urls.csv"

df = pd.read_csv(csv_path, nrows=5, encoding="utf-8")

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.to_string(index=False))

print("\nData types:")
print(df.dtypes)

print("\nFile size:", f"{csv_path.stat().st_size:,}", "bytes")
