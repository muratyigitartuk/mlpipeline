import pytest
from app import create_app
import jwt
from datetime import datetime, timedelta, timezone

@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def generate_valid_token():
    """Generate a valid JWT token for testing"""
    secret_key = "default-secret"  # Must match the jwt_secret in config.yaml
    algorithm = "HS256"
    payload = {
        "roles": ["admin"],  # Must match allowed_roles in config.yaml
        "exp": datetime.now(timezone.utc) + timedelta(minutes=60),  # 60 minutes valid
        "iat": datetime.now(timezone.utc),  # Issuance time
        "sub": "test-user"
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)

def test_health_endpoint(client):
    """Test the /api/v1/health endpoint"""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_predict_endpoint_with_valid_token(client):
    """Test the /api/v1/predict endpoint with a valid JWT token"""
    valid_token = generate_valid_token()  # Dynamically generate a new token
    headers = {'Authorization': f'Bearer {valid_token}'}
    data = {'data': [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]}
    response = client.post('/api/v1/predict', json=data, headers=headers)
    assert response.status_code == 200
    assert 'predictions' in response.json
