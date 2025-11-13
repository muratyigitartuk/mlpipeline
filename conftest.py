import pytest
import yaml
import numpy as np
import sys
from pathlib import Path

# Ensure the directory containing app.py is importable as 'app'
here = Path(__file__).resolve().parent
sys.path.append(str(here))
mp = here / 'ml-pipeline'
if mp.exists():
    sys.path.append(str(mp))

from app import create_app, AppConfig, ModelConfig, SecurityConfig

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file"""
    config = {
        "model": {
            "model_path": "model/model.joblib",
            "n_features": 10,
            "input_validation": {}
        },
        "security": {
            "jwt_secret": "test_secret_key_minimum_32_chars_long_for_testing",
            "jwt_algorithm": "HS256",
            "token_expire_minutes": 30
        },
        "allowed_roles": ["admin", "user"],
        "rate_limit": 100,
        "nan_replacement": 0.0,
        "enable_auth": True
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding='utf-8') as f:
        yaml.dump(config, f)
    return config_path

@pytest.fixture
def mock_model():
    """Mock ML model for testing"""
    class MockModel:
        def predict(self, input_data):
            return np.zeros(len(input_data))
    return MockModel()

@pytest.fixture
def mock_pipeline(mock_model):
    """Mock MLPipeline for testing"""
    from app import MLPipeline
    pipeline = MLPipeline()
    pipeline.model = mock_model
    pipeline._health_status = {'status': 'ready'}
    pipeline._config = AppConfig(
        model=ModelConfig(model_path="model/model.joblib", n_features=10, input_validation={}),
        security=SecurityConfig(jwt_secret="test_secret_key_minimum_32_chars_long_for_testing",
                                jwt_algorithm="HS256", token_expire_minutes=30),
        allowed_roles=["admin", "user"],
        rate_limit=100,
        nan_replacement=0.0,
        enable_auth=True,
        ssl_cert=None,
        ssl_key=None
    )
    return pipeline

@pytest.fixture
def client():
    """Flask test client"""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
