
from feature_extractor import extract_features
from predict import predict_features

examples = [
    "https://www.wikipedia.org",
    "http://192.168.1.1/login",
    "https://paypal.com@evil-example.com/login",
]

for url in examples:
    print("\nURL:", url)

    try:
        features = extract_features(url)
        result = predict_features(features)

        print("Extracted features:", features)
        print("ML prediction:", result["verdict"])
        print("Predicted label:", result["predicted_label"])
        print("Risk score:", result["risk_score"], "/ 100")
        print("Note:", result["note"])

    except (ValueError, FileNotFoundError) as error:
        print("Error:", error)
