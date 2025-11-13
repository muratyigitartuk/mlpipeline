import os
from pathlib import Path
import jwt
import numpy as np
from pipeline.training.train import train_logistic
import app as app_module

def test_predict_endpoint_e2e(tmp_path):
    os.environ['CONFIG_PATH'] = str(Path('ml-pipeline') / 'config.yaml')
    os.environ['APP_ENV'] = 'dev'
    feats = tmp_path / 'feats.parquet'
    import pandas as pd
    df = pd.DataFrame({f'f{i}': np.random.rand(5) for i in range(10)})
    df.to_parquet(feats, index=False)
    model_out = Path('ml-pipeline/model/production/model.joblib')
    model_out.parent.mkdir(parents=True, exist_ok=True)
    train_logistic(str(feats), str(model_out), seed=44)
    app = app_module.create_app()
    app.config['TESTING'] = True
    client = app.test_client()
    token = jwt.encode({'roles': ['admin']}, 'default-secret', algorithm='HS256')
    headers = {'Authorization': f'Bearer {token}'}
    payload = {'data': [[float(i) for i in range(10)]]}
    resp = client.post('/api/v1/predict', json=payload, headers=headers)
    assert resp.status_code == 200
    assert 'predictions' in resp.json
