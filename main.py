from fastapi import FastAPI 
from pydantic import BaseModel 
import joblib 

MODEL_PATH = "category_model.joblib"

app = FastAPI(
    title="Job Market Category Predictor",
    description="Predicts a job category(data analyst, data scientist, data engineer, etc.) from a job title and description, trained on real Irish job market data.",
)

model = joblib.load(MODEL_PATH)

class JobInput(BaseModel):
    title: str 
    snippet: str = ""

class PredictionOutput(BaseModel):
    predicted_category: str 

@app.get("/")
def root():
    return { 
        "message": "Job Market Category Predictor API",
        "usage": "POST to /predict with {'title': '...', 'snippet': '...'}",
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput):
    text = f"{job.title} {job.snippet}"
    prediction = model.predict([text])[0]
    return PredictionOutput(predicted_category=prediction)