# export_model_to_pkl.py
import os
import mlflow
import mlflow.sklearn
import joblib

MODEL_NAME = "best_fraud_model"
MODEL_ALIAS = "champion"

model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

print(f"Loading model from MLflow URI: {model_uri}")

sk_model = mlflow.sklearn.load_model(model_uri)

print("Loaded model:", type(sk_model))

# Save as plain pickle using joblib
joblib.dump(sk_model, "model.pkl")

print("Saved model.pkl in current directory")