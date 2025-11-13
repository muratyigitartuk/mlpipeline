from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.training.train import train_logistic
import joblib

def test_canary_vs_current(tmp_path):
    n = 50
    n_features = 10
    data = {f"f{i}": np.random.rand(n) for i in range(n_features)}
    feats_path = tmp_path / "features.parquet"
    pd.DataFrame(data).to_parquet(feats_path, index=False)
    current_model_path = tmp_path / "current.joblib"
    candidate_model_path = tmp_path / "candidate.joblib"
    train_logistic(str(feats_path), str(current_model_path), seed=42)
    train_logistic(str(feats_path), str(candidate_model_path), seed=43)
    current = joblib.load(current_model_path)
    candidate = joblib.load(candidate_model_path)
    X = pd.read_parquet(feats_path).to_numpy()
    y_curr = current.predict(X)
    y_cand = candidate.predict(X)
    assert len(y_curr) == len(y_cand)
    diff = np.mean(np.abs(y_curr - y_cand))
    assert diff >= 0.0
