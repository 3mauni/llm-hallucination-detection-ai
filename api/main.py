from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="LLM Hallucination Detection API"
)

model = joblib.load("models/tfidf_logistic_model.pkl")

class PredictionRequest(BaseModel):
    prompt: str
    response: str

@app.get("/")
def home():
    return {"message": "LLM Hallucination Detection API Running"}

@app.post("/predict")
def predict(data: PredictionRequest):

    text = data.prompt + " " + data.response

    prediction = model.predict([text])[0]
    probability = model.predict_proba([text])[0]

    return {
        "hallucination_prediction": int(prediction),
        "factual_probability": float(probability[0]),
        "hallucination_probability": float(probability[1])
    }