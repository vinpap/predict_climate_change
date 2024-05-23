"""
This script monitors the model performance.
"""

import os
import yaml
import psycopg2
import sys
import argparse
import smtplib
from email.mime.text import MIMEText
from datetime import date

import requests
import pandas as pd

API_TOKEN = os.environ["API_TOKEN"]
SMTP_LOGIN = os.environ["SMTP_LOGIN"]
SMTP_PASSWORD = os.environ["SMTP_PASSWORD"]
try:
    DB_PWD = os.environ["DB_PASSWORD"]
except KeyError:
    print(
        "NOTE: DB_PASSWORD has not been set in the environment variables. This is only a problem if you are loading monitoring data from a PostgreSQL database"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Temperature prediction monitoring script",
        description="This script is used to monitor the temperature prediction model by making sure its performance is still viable",
    )
    parser.add_argument(
        "--setup",
        "-s",
        action="store_true",
        help="Trains the model through the API for the first time and defines the performance threshold",
    )
    return parser.parse_args()


def should_run() -> bool:
    """
    Compares the current date with the last monitoring date stored in
    monitoring_cfg.yml. Returns True if more than a month has passed, False
    otherwise.
    """
    with open("./monitoring_cfg.yml", "r") as config_file:
        cfg = yaml.safe_load(config_file)
        last_running_date = cfg["LAST_RUNNING_DATE"]
    current_date = date.today()
    time_diff = current_date - last_running_date
    return time_diff.days >= 30


def load_data(config: dict):
    """
    Loads the data that should be used to perform the monitoring. The data source
    is specified in monitoring_cfg.yml.
    """
    if config["DATA"]["LOAD_FROM_CSV"]:
        data = load_from_csv(config["DATA"]["CSV_PATH"])

    else:
        data = load_from_postgresql(config)

    return data


def load_from_csv(filepath: str) -> pd.DataFrame:
    """
    Loads the data from a CSV file.

    The data provided must contain one column named "measurement_dates" and
    another one named "temperatures".
    """

    data = pd.read_csv(filepath)
    filtered_data = data[["measurement_dates", "temperatures"]]
    filtered_data["measurement_dates"] = pd.to_datetime(
        filtered_data["measurement_dates"]
    )
    filtered_data["measurement_dates"] = filtered_data["measurement_dates"].dt.strftime(
        "%m/%Y"
    )
    return filtered_data


def load_from_postgresql(config: dict) -> pd.DataFrame:
    """
    Loads the data from the database specified in the configuration file.

    This function looks for a table named "monitoring" in that database and loads
    the columns named "measurement_dates" and "temperatures".
    """

    data = {"measurement_dates": [], "temperatures": []}

    conn = psycopg2.connect(
        database=config["DATA"]["DB_NAME"],
        host=config["DATA"]["DB_URL"],
        user=config["DATA"]["DB_USER"],
        password=DB_PWD,
        port=config["DATA"]["DB_PORT"],
    )
    cursor = conn.cursor()
    retrieval_request = "SELECT measurement_dates, temperatures FROM monitoring"
    cursor.execute(retrieval_request)
    results = cursor.fetchall()

    for record in results:
        data["measurement_dates"].append(record[0])
        data["temperatures"].append(record[1])

    df = pd.DataFrame(data)
    df["measurement_dates"] = pd.to_datetime(df["measurement_dates"])
    df["measurement_dates"] = df["measurement_dates"].dt.strftime("%m/%Y")
    return df


def send_email(subject: str, body: str):
    """
    Sends an email to the recipient specified in the config file.
    """
    with open("./monitoring_cfg.yml", "r") as config_file:
        cfg = yaml.safe_load(config_file)
        sender = SMTP_LOGIN
        recipient = [cfg["EMAIL_SETTINGS"]["EMAIL_RECIPIENT"]]
        pwd = SMTP_PASSWORD
        smtp_server = cfg["EMAIL_SETTINGS"]["SMTP_SERVER"]

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipient)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp_server:
            smtp_server.login(sender, SMTP_PASSWORD)
            smtp_server.sendmail(
                "papelardvincent@gmail.com", recipient, msg.as_string()
            )
            print("Message sent!")


def split_dataset(data: pd.DataFrame) -> tuple:
    """
    Splits, the dataframe provided into a train and test datasets.
    """
    df_size = len(data)
    train_ratio = 0.8
    splitting_index = round(train_ratio * df_size)

    train = data.iloc[:splitting_index, :]
    test = data.iloc[splitting_index:, :]

    return train, test


def setup():
    """
    Trains the model through the API and computes its root mean squared error.
    Stores the result as a threshold for future monitoring.
    """
    with open("./monitoring_cfg.yml", "r") as config_file:
        cfg = yaml.safe_load(config_file)
        data = load_data(cfg)

        # Splitting the data in train and test datasets
        train, test = split_dataset(data)

        # Training the temperature prediction model through the API
        request_data = {
            "start_date": train["measurement_dates"].iloc[0],
            "temperatures": list(train["temperatures"]),
            "secret_token": API_TOKEN,
        }

        response = requests.post(url=cfg["API_ENDPOINT"] + "train", json=request_data)
        if response.status_code < 200 or response.status_code >= 300:
            raise RuntimeError(
                "Unable to use the API, please make sure it is up and running at the URL specified in monitoring_cfg.yml"
            )

        # Testing the model and saving 105% of the RMSE as the new performance threshold
        test_request_data = {
            "start_date": test["measurement_dates"].iloc[0],
            "temperatures": list(test["temperatures"]),
            "secret_token": API_TOKEN,
        }

        response = requests.post(
            url=cfg["API_ENDPOINT"] + "test", json=test_request_data
        )
        new_threshold = 1.05 * response.json()["RMSE"]

        cfg["RMSE_THRESHOLD"] = new_threshold

    with open("./monitoring_cfg.yml", "w") as config_file:
        yaml.dump(cfg, config_file)


def monitor():
    """
    Monitors the model's current performance and alerts the user if it should be
    retrained.
    """

    with open("./monitoring_cfg.yml") as config_file:
        cfg = yaml.safe_load(config_file)
        if "RMSE_THRESHOLD" not in cfg or cfg["RMSE_THRESHOLD"] == -1:
            print(
                "No performance threshold found in the configuration file. Please run this script with the --setup flag at least once before using it to monitor the prediction model"
            )
            sys.exit(0)
        data = load_data(cfg)
        threshold = cfg["RMSE_THRESHOLD"]

        test_request_data = {
            "start_date": data["measurement_dates"].iloc[0],
            "temperatures": list(data["temperatures"]),
            "secret_token": API_TOKEN,
        }

        response = requests.post(
            url=cfg["API_ENDPOINT"] + "test", json=test_request_data
        )
        rmse = response.json()["RMSE"]

        if rmse > threshold:
            print(
                f"The model performance went above the root mean squared error set at {threshold}. Now retraining the model"
            )

            train, test = split_dataset(data)
            # Training the temperature prediction model through the API
            request_data = {
                "start_date": train["measurement_dates"].iloc[0],
                "temperatures": list(train["temperatures"]),
                "secret_token": API_TOKEN,
            }

            response = requests.post(
                url=cfg["API_ENDPOINT"] + "train", json=request_data
            )
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(
                    "Unable to use the API, please make sure it is up and running at the URL specified in monitoring_cfg.yml"
                )

            # Testing the model
            test_request_data = {
                "start_date": test["measurement_dates"].iloc[0],
                "temperatures": list(test["temperatures"]),
                "secret_token": API_TOKEN,
            }

            response = requests.post(
                url=cfg["API_ENDPOINT"] + "test", json=test_request_data
            )
            rmse = response.json()["RMSE"]

            if rmse > threshold:
                print(
                    f"RMSE is still too high after retraining the model ({rmse} compared to a threshold set at {threshold}). Sending an email alert to {cfg['EMAIL_SETTINGS']['EMAIL_RECIPIENT']}"
                )
                subject = "ATTENTION REQUIRED - your model's performance is getting bad"
                message = f"This is an automated email, please do not reply.\nThe AI model deployed at {cfg['API_ENDPOINT']} had to be retrained because its error metric went above the specified Root Mean Square Error set at {threshold}.\nUnfortunately, retraining the model did not enable it to perform well enough.\nThe new RMSE is {rmse}, but the threshold is set at {threshold}. Further analysis is advised in order to get the model back on track."

            else:
                print("The model's performance went back below the threshold")
                subject = "Your model has been retrained"
                message = f"This is an automated email, please do not reply.\nThe AI model deployed at {cfg['API_ENDPOINT']} had to be retrained because its error metric went above the specified Root Mean Square Error set at {threshold}.\nRetraining allowed the model to get a new RMSE set at {rmse}, which matches the threshold."

            send_email(subject, message)


if __name__ == "__main__":
    args = parse_args()

    if args.setup:
        setup()
    else:
        if not should_run():
            sys.exit(0)
        monitor()

    with open("./monitoring_cfg.yml") as config_file:
        cfg = yaml.safe_load(config_file)
        cfg["LAST_RUNNING_DATE"] = date.today()
    with open("./monitoring_cfg.yml", "w") as config_file:
        yaml.dump(cfg, config_file)
