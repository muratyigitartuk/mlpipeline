from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def train_logistic(features_path: str, model_out: str, seed: int = 42) -> str:
    df = pd.read_parquet(features_path)
    X = df.to_numpy()
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=X.shape[0])
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    out_dir = Path(model_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    return model_out

def train_with_mlflow(features_path: str, model_out: str, tracking_uri: str = None, experiment: str = None) -> str:
    df = pd.read_parquet(features_path)
    X = df.to_numpy()
    y = np.zeros(X.shape[0])
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=100)
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        if experiment:
            mlflow.set_experiment(experiment)
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "init_model")
    model.fit(X, y)
    out_dir = Path(model_out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    return model_out
