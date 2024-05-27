import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="../output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc_taxi_experiment")
    #mlflow.log_artifact("/mlruns")
    mlflow.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():

        mlflow.set_tag("developer", "agustin")
        mlflow.set_tag("autolog", "yes")

        mlflow.log_param("train-data", "./output/train.pkl")
        mlflow.log_param("valid-data", "./output/val.pkl")

        max_depth = 20
        #mlflow.log_param("max_depth", max_depth)
        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred)
        mlflow.log_param("rmse", rmse)


if __name__ == '__main__':
    run_train()
