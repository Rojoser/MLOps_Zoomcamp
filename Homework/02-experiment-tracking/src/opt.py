import os
import pickle
import click
import mlflow
import optuna

from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
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
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("random-forest-optuna")

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50, step=1),
            'max_depth': trial.suggest_int('max_depth', 1, 20, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, step=1),
            'random_state': 42,
            'n_jobs': -1
        }
    
        with mlflow.start_run():
            mlflow.set_tag("developer", "agustin")
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_param("rmse", rmse)
            mlflow.log_param("n_estimators",params['n_estimators'])
            mlflow.log_param("max_depth",params['max_depth'])
            mlflow.log_param("min_samples_split",params['min_samples_split'])
            mlflow.log_param("min_samples_leaf",params['min_samples_leaf'])

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


if __name__ == '__main__':
    run_optimization()
