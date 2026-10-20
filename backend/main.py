from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("heart_disease_model.pkl")


class HeartPredictionInput(BaseModel):
    age: int
    sex: str
    bmi: float
    smoking_status: str
    alcohol_intake: str
    physical_activity: str
    diet_quality: str
    sleep_hours: float
    stress_level: int
    diabetes: int
    hypertension: int
    family_history: int
    previous_heart_event: int
    resting_bp: float
    cholesterol: float
    fasting_blood_sugar: float
    max_heart_rate: float
    resting_ecg: str
    exercise_angina: int
    st_depression: float
    st_slope: str
    avg_resting_hr: float
    hrv_score: float


@app.get("/")
def home():
    return {"Heart Disease Detector Health Checkup : Healthy"}


@app.post("/predict")
def predict(data: HeartPredictionInput):

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    confidence = probabilities[prediction] * 100

    return {
        "prediction": int(prediction),
        "result": (
            "Heart disease detected" if prediction == 1 else "No heart disease detected"
        ),
        "confidence": round(float(confidence), 2),
    }
