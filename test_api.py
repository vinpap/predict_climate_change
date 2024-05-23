"""
This script is used to make sure the API is working properly.
"""
import requests
import pandas as pd


def test_train():
    """
    Tests the /train endpoint.
    """

    data = pd.read_csv("./data/GlobalTemperatures.csv")
    temperatures = data["LandAndOceanAverageTemperature"].interpolate(
        limit_direction="backward"
    )[1300:]
    dates = pd.to_datetime(data["dt"])[1300:]

    data = {"start_date": str(dates.iloc[0]), "temperatures": list(temperatures)}
    response = requests.post("http://127.0.0.1:8000/train", json=data)

    assert response.status_code == 200
