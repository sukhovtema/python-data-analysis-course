from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.pkl")


# POST модель ввода
class PropertyFeatures(BaseModel):
    total_square: float
    rooms: int
    floor: int


# health-check
@app.get("/health")
def health_check():
    return {"status": "ok"}


# GET-запрос: /predict_get?total_square=50&rooms=2&floor=3
@app.get("/predict_get")
def predict_get(total_square: float = Query(...), rooms: int = Query(...), floor: int = Query(...)):
    X = np.array([[total_square, rooms, floor]])
    prediction = model.predict(X)[0]
    return {"prediction": round(prediction, 2)}


# POST-запрос: /predict_post с JSON телом
@app.post("/predict_post")
def predict_post(features: PropertyFeatures):
    X = np.array([[features.total_square, features.rooms, features.floor]])
    prediction = model.predict(X)[0]
    return {"prediction": round(prediction, 2)}