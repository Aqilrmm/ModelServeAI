import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "ModelServeAI is running!" in response.json()["message"]

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    test_data = {
        "user_id": 123,
        "age": 25,
        "gender": "M",
        "category_preference": "Electronics",
        "price_range": "Medium"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "recommendations" in result
    assert "confidence" in result
    assert result["user_id"] == 123

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
