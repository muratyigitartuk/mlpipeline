# create_model.py
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
from pathlib import Path

X_train = np.random.rand(100, 10)  # 100 samples, 10 features (compatible with n_features: 10 in config.yaml)
y_train = np.random.randint(0, 2, 100)  # Classes 0 or 1
model = LogisticRegression()
model.fit(X_train, y_train)

model_dir = Path(__file__).resolve().parent / "model"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "model.joblib"
joblib.dump(model, str(model_path))
print(str(model_path))
