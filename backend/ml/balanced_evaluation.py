
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
CSV = BASE / "data" / "phishing_urls.csv"
MODEL = BASE / "models" / "phishing_model.pth"

FEATURES = [
    "url_length", "num_dots", "has_https", "has_ip",
    "num_subdirs", "num_params", "suspicious_words",
    "special_char_count", "digits_count", "entropy",
]

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv(CSV)
df = df[FEATURES + ["label"]].dropna()
df["label"] = df["label"].astype(int)
df = df.drop_duplicates(subset=FEATURES + ["label"])

# Keep all legitimate examples and sample an equal number of phishing examples.
legitimate = df[df["label"] == 0]
phishing = df[df["label"] == 1].sample(
    n=len(legitimate), random_state=42
)
balanced = pd.concat([legitimate, phishing]).sample(
    frac=1, random_state=42
)

X = balanced[FEATURES].to_numpy(dtype=np.float32)
y = balanced["label"].to_numpy(dtype=np.int64)

# Split balanced data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Use the exact scaler saved during model training.
scaler_data = np.load(BASE / "models" / "scaler.npz")
saved_features = scaler_data["features"].tolist()

if saved_features != FEATURES:
    raise ValueError("Feature order does not match saved scaler.")

X_train = (
    (X_train - scaler_data["mean"]) / scaler_data["scale"]
).astype(np.float32)

X_test = (
    (X_test - scaler_data["mean"]) / scaler_data["scale"]
).astype(np.float32)

class PhishingNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)

# Load the trained weights.
checkpoint = torch.load(
    MODEL, map_location="cpu", weights_only=True
)
model = PhishingNet(len(FEATURES))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

with torch.no_grad():
    probabilities = torch.softmax(
        model(torch.from_numpy(X_test)), dim=1
    )
    predictions = probabilities.argmax(dim=1).numpy()

print("Balanced test set:", len(y_test))
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["legitimate (0)", "phishing (1)"],
    zero_division=0,
))
print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
