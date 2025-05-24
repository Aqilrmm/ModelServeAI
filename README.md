# 🧠 ModelServeAI

Professional AI Model Serving System with MLflow tracking and Prometheus monitoring.

## 🚀 Features

- **FastAPI REST API** - High-performance model serving
- **MLflow Integration** - Experiment tracking and model versioning  
- **Prometheus + Grafana** - Real-time monitoring and metrics
- **Docker Support** - Easy deployment and scaling
- **CI/CD Pipeline** - Automated testing and deployment
- **Health Checks** - System reliability monitoring

## 📋 Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

## 🛠️ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Aqilrmm/ModelServeAI.git
cd ModelServeAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Model
```bash
python app/predict.py
```

### 4. Run Application
```bash
# Development
uvicorn app.main:app --reload --port 8000

# Production with Docker Compose
docker-compose up -d
```

## 🔗 Endpoints

- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## 📊 API Usage

### Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "age": 25,
    "gender": "M", 
    "category_preference": "Electronics",
    "price_range": "Medium"
  }'
```

### Response
```json
{
  "user_id": 123,
  "recommendations": [
    {
      "product_id": "product_1",
      "score": 0.85
    }
  ],
  "confidence": 0.85,
  "timestamp": "1640995200.0"
}
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t modelserveai .

# Run container
docker run -p 8000:8000 modelserveai

# Full stack with monitoring
docker-compose up -d
```

## 📈 Monitoring

### Grafana Dashboard
1. Access Grafana at http://localhost:3000
2. Login: admin/admin
3. Import dashboard from `monitoring/grafana_dashboard.json`

### Available Metrics
- Request rate and latency
- Model prediction success rate  
- System health status
- MLflow experiment tracking

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

## 🔄 CI/CD

GitHub Actions pipeline includes:
- Automated testing
- Docker image building
- Model training validation
- Deployment automation

## 📁 Project Structure

```
ModelServeAI/
├── app/
│   ├── main.py         # FastAPI application
│   ├── predict.py      # Model training logic
│   ├── utils.py        # Helper functions
│   └── model.pkl       # Trained model
├── tests/              # Test suite
├── monitoring/         # Prometheus & Grafana configs
├── .github/workflows/  # CI/CD pipeline
├── Dockerfile         # Container configuration
├── docker-compose.yml # Multi-service setup
└── requirements.txt   # Dependencies
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**ModelServeAI** - Production-ready AI model serving made simple! 🎯