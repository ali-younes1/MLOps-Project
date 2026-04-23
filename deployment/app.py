import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import warnings
import os

warnings.filterwarnings("ignore")

app = FastAPI()

print("Loading model from model.pkl...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)
print("Model loaded:", type(model))

FEATURE_NAMES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        if len(input_data.features) != 30:
            return {"error": f"Expected 30 features, got {len(input_data.features)}"}

        df_input = pd.DataFrame([input_data.features], columns=FEATURE_NAMES)
        prediction = model.predict(df_input)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        return {"error": str(e)}