import os
import pickle
from typing import Sequence, List

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pmdarima.arima import ARIMA

class WeatherData(BaseModel):
    """
    Data structure that stores average monthly temperatures for a given period of time.
    """
    start_date: str
    temperatures: List[float]

def get_model():
    """
    Loads the SARIMA model.

    if the model was already trained, this function loads it from file. Otherwise
    it creates, trains and saves the model using a default dataset located in the 'data'
    subfolder.
    """
    model_path = "./model.pkl"
    if os.path.exists(model_path) and os.path.isfile(model_path):
        print(f"Loading SARIMA model from {model_path}")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
            return model
    
    print(f"Training SARIMA model")
    model = ARIMA(
        order=(1, 1, 1),
        seasonal_order=(2, 0, 1, 12)
                )
    data = pd.read_csv("./data/GlobalTemperatures.csv")
    temperature = data["LandAndOceanAverageTemperature"].interpolate(limit_direction="backward")[1300:]
    model.fit(temperature)
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    
    return model


model = get_model()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to our API.\
             Please use one of these endpoints: /train, /test, /predict"}

@app.post("/train")
async def train(data: WeatherData):
    """
    Trains the model using the temperatures provided.
    """
    model.fit(data.temperatures)
    return {"message": "OK"}

@app.get("/predict")
async def predict():
    return {"message": "Hello World"}

@app.get("/test")
async def test(data: WeatherData):
    return {"message": "Hello World"}