import os
import pickle
import click
import mlflow

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.metrics import mean_squared_error

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="../output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_optimization(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("random-forest-hyperopt")

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    def objective(params):
    
        with mlflow.start_run():
            mlflow.set_tag("developer", "agustin")
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, "validaton")],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100,1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objectve": "reg:linear",
        "seed": 42,
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )
  


if __name__ == '__main__':
    run_optimization()
