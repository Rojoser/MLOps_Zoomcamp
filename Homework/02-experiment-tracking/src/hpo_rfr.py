import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

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

    def objective(params):
    
        with mlflow.start_run():
            mlflow.set_tag("developer", "agustin")
            mlflow.set_tag("model", "rf-regressor")
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_param("n_estimators",params['n_estimators'])
            mlflow.log_param("max_depth",params['max_depth'])
            mlflow.log_param("min_samples_split",params['min_samples_split'])
            mlflow.log_param("min_samples_leaf",params['min_samples_leaf'])

        return {"loss": rmse, "status": STATUS_OK}
    
    search_space = {
        'n_estimators': scope.int(hp.quniform("n_estimators", 1, 100,1)),
        'max_depth': scope.int(hp.quniform("max_depth", 3, 20,1)),
        'min_samples_split': scope.int(hp.quniform("min_samples_split", 2, 20,1)),
        'min_samples_leaf': scope.int(hp.quniform("min_samples_leaf", 2, 20,1)), 
        'random_state': 42
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
