import os
import pickle
import math
from typing import Sequence, List

import pandas as pd
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_squared_error

class WeatherData(BaseModel):
    """
    Data structure that stores average monthly temperatures for a given period of time.
    """
    start_date: str
    temperatures: List[float]
    secret_token: str

class ForecastDate(BaseModel):
    """
    Stores a single date.
    """
    date: str
    secret_token: str


def get_model():
    """
    Loads the SARIMA model.

    if the model was already trained, this function loads it from file. Otherwise
    it returns False.
    """
    model_path = "./model.pkl"
    dates_path = "./dates.pkl"
    if os.path.exists(model_path) and os.path.isfile(model_path) and os.path.exists(dates_path) and os.path.isfile(dates_path):
        print(f"Loading SARIMA model from {model_path}")
        with open(model_path, "rb") as model_file, open(dates_path, "rb") as dates_file:
            model = pickle.load(model_file)
            dates = pickle.load(dates_file)
            return model, dates
    
    else:
        return False, None
    

# model_dates stores the range of dates covered by the trained model
model, model_dates = get_model()
app = FastAPI()
security_token = os.environ["API_TOKEN"]

@app.post("/", status_code=200)
async def root(secret_token: str, response: Response):
    if secret_token != security_token:
        return JSONResponse(
            status_code=401, content={"msg": "Your security code is not valid"}
        )
    return {"message": "Welcome to our API.\
             Please use one of these endpoints: /train, /test, /predict"}

@app.post("/train", status_code=200)
async def train(data: WeatherData):
    """
    Trains the model using the temperatures provided.
    """
    global model
    global model_dates

    if data.secret_token != security_token:
        return JSONResponse(
            status_code=401, content={"msg": "Your security code is not valid"}
        )

    try:
        start_date = pd.to_datetime(data.start_date, format="%m/%Y")
    except ValueError:
        return {"message": "Please provide the temperature series start date in format mm/yyyy"}
    
    # The hyperparameters below are those that provide the best performance
    # on monthly temperature series. See report for more info
    model = ARIMA(
        order=(1, 1, 1),
        seasonal_order=(2, 0, 1, 12)
                )
    temperature = data.temperatures
    model.fit(temperature)

    # Updating model dates
    model_dates = []
    for i in range(len(temperature)):
        model_dates.append(start_date + pd.DateOffset(months=i))

    model_path = "./model.pkl"
    dates_path = "./dates.pkl"
    with open(model_path, "wb") as model_file, open(dates_path, "wb") as dates_file:
        pickle.dump(model, model_file)
        pickle.dump(model_dates, dates_file)


    return {"message": "OK"}

@app.post("/predict", status_code=200)
async def predict(forecast_date: ForecastDate):

    if forecast_date.secret_token != security_token:
        return JSONResponse(
            status_code=401, content={"msg": "Your security code is not valid"}
        )
    if not model:
        return {"message": "No model has been trained. Please train the model first by sending monthly temperature series to the /train endpoint"}
    
    try:
        forecast_date = pd.to_datetime(forecast_date.date, format="%m/%Y")
    except ValueError:
        return {"message": "Please provide the forecast date in format mm/yyyy"}
    
    if forecast_date <= model_dates[-1]:
        return {"message": f"The date you entered is older than the last date stored by the forecast model, which is {str(model_dates[-1])}. Please enter a date after this one."}

    # Storing all the dates between the model's last known date and the requested date
    months_delta = forecast_date.to_period("M") - model_dates[-1].to_period("M")
    months_covered = months_delta.n + 1

    months = []
    for month in range(1, months_covered):
        months.append(model_dates[-1] + pd.DateOffset(months=month))

    predicted_temperatures = list(model.predict(months_covered-1))

    return {"dates": months, "temperatures": predicted_temperatures}

@app.post("/test", status_code=200)
async def test(data: WeatherData):

    if data.secret_token != security_token:
        return JSONResponse(
            status_code=401, content={"msg": "Your security code is not valid"}
        )
    try:
        start_date = pd.to_datetime(data.start_date, format="%m/%Y")
    except ValueError:
        return {"message": "Please provide the start date in format mm/yyyy"}
    
    if not model:
        return {"message": "No model has been trained. Please train the model first by sending monthly temperature series to the /train endpoint"}
    
    test_temp = data.temperatures
    # Any test data older than the model's last registered date cannot be used and need to be removed
    if start_date <= model_dates[-1]:
        months_delta = model_dates[-1].to_period("M") - start_date.to_period("M")
        months_diff = months_delta.n
        if len(test_temp) <= months_diff:
            return {"message": "The test data you sent does not include any unknown data, therefore it is impossible to test the model with it"}
        test_temp = test_temp[months_diff+1:]
    
    pred_temp = model.predict(len(test_temp))
    rmse = math.sqrt(mean_squared_error(test_temp, pred_temp))
    
    return {"RMSE": rmse}