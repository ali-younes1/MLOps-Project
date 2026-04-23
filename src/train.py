import json
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create output folders if they do not already exist
Path("outputs").mkdir(parents=True, exist_ok=True)
Path("registry").mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv('data/sample/creditcard_sample.csv')
X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = [
    (
        "Logistic Regression",
        LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            random_state=13
        )
    ),
    (
        "KNN (k=3)",
        KNeighborsClassifier(
            n_neighbors=3
        )
    ),
]

best_model = None
best_model_name = None
best_metrics = None
best_f1 = -1

# 3. Train models
for model_name, model in models:
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = model_name
        best_metrics = {
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4)
        }

model = best_model
metrics = best_metrics

# 5. Save model
joblib.dump(model, "outputs/model.pkl")

# 6. Save metrics
with open("outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 7. Simulate a simple registry
registry_path = "registry/champion.json"

new_candidate = {
    "model_name": best_model_name,
    "model_path": "outputs/model.pkl",
    "metrics_path": "outputs/metrics.json",
    "accuracy": metrics["accuracy"],
    "f1_score": metrics["f1_score"],
    "status": "candidate"
}

print(new_candidate)

if os.path.exists(registry_path):
    with open(registry_path, "r") as f:
        current_champion = json.load(f)

    # Replace champion only if the new model has better F1
    if new_candidate["f1_score"] > current_champion.get("f1_score", 0):
        new_candidate["status"] = "champion"

        with open(registry_path, "w") as f:
            json.dump(new_candidate, f, indent=4)

        print("New model promoted to champion.")
    else:
        print("Current champion retained.")

else:
    new_candidate["status"] = "champion"

    with open(registry_path, "w") as f:
        json.dump(new_candidate, f, indent=4)

    print("No champion existed. New model set as champion.")

print("Training complete.")
print("Metrics:", metrics)