# API Directory - FastAPI Service

Production-ready REST API for pod forecasting predictions. Built with FastAPI, includes monitoring, logging, and comprehensive error handling.

---

## Overview

**FastAPI service that:**
- Loads trained ML model
- Serves predictions via REST endpoints
- Provides interactive documentation
- Exports Prometheus metrics
- Includes health checks
- Logs all requests

**Tech Stack:**
- FastAPI (async, type-safe)
- Pydantic (validation)
- Prometheus (metrics)
- uvicorn (ASGI server)

---

## Files

```
api/
├── main.py              # FastAPI application
├── predictor.py         # ML prediction logic
├── models.py            # Pydantic schemas
├── requirements.txt     # Dependencies
└── logs/               # Application logs
    └── pod_api.log     # Request/response logs
```

---

## File Descriptions

### main.py

**FastAPI application with all endpoints**

**Features:**
- Request tracing (X-Request-ID)
- Prometheus metrics
- Exception handling
- Health checks
- CORS support
- Structured logging

**Endpoints:**
```python
GET  /              # API information
GET  /health        # Health check
GET  /metrics       # Prometheus metrics
POST /predict       # Single prediction
POST /forecast/batch # Batch predictions
GET  /model-info    # Model metadata
GET  /docs          # Interactive docs
```

---

### predictor.py

**ML prediction logic**

**Responsibilities:**
- Load trained model
- Engineer features
- Make predictions
- Calculate confidence intervals

**Key Functions:**
```python
class PodPredictor:
    __init__()              # Load model + scaler
    engineer_features()     # Create 10 features
    predict()               # Make prediction
    get_model_info()        # Model metadata
```

---

### models.py

**Pydantic data models**

**Schemas:**
- `PredictionRequest` - Single prediction input
- `PredictionResponse` - Single prediction output
- `BatchPredictionRequest` - Batch input
- `BatchPredictionResponse` - Batch output
- `HealthResponse` - Health check response

**Features:**
- Type validation
- Multi-format date support (DD/MM/YYYY, YYYY-MM-DD, MM/DD/YYYY)
- Value range validation
- Auto-generated examples

---

### requirements.txt

**Python dependencies**

```txt
# API Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# ML
scikit-learn==1.3.0
joblib==1.3.2
numpy==1.24.3

# Monitoring
prometheus-client==0.19.0

# Logging
python-json-logger==2.0.7

# Utilities
python-dotenv==1.0.0
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2. Ensure Model Exists

```bash
# Check model files
ls -lh ../models/model.pkl ../models/scaler.pkl

# If missing, train model
cd ../scripts
python train_model.py
cd ../api
```

### 3. Start API

```bash
uvicorn main:app --reload --port 5000
```

**Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:5000
INFO:     Application startup complete.
INFO:     Model loaded from ../models/model.pkl
INFO:     Scaler loaded from ../models/scaler.pkl
```

### 4. Test API

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-07-15",
    "gmv": 9500000,
    "users": 85000,
    "marketing_cost": 175000
  }'
```

### 5. View Documentation

Open browser: http://localhost:5000/docs

---

## API Endpoints

### GET /

**API Information**

**Response:**
```json
{
  "name": "Pod Forecasting API",
  "version": "1.0.0",
  "description": "ML-powered Kubernetes pod forecasting",
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict": "POST /predict",
    "batch": "POST /forecast/batch"
  }
}
```

---

### GET /health

**Health Check (Kubernetes probes)**

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "RandomForestRegressor",
    "features": ["gmv", "users", "marketing_cost", ...],
    "n_features": 10
  }
}
```

**Status Codes:**
- `200` - Healthy
- `503` - Model not loaded

**Usage:**
```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
```

---

### GET /metrics

**Prometheus Metrics**

**Response:** Prometheus text format

**Sample Metrics:**
```prometheus
# HELP pod_api_requests_total Total API requests
# TYPE pod_api_requests_total counter
pod_api_requests_total{method="POST",endpoint="/predict",status="200"} 142

# HELP pod_api_request_duration_seconds Request duration
# TYPE pod_api_request_duration_seconds histogram
pod_api_request_duration_seconds_bucket{endpoint="/predict",le="0.05"} 120
pod_api_request_duration_seconds_count{endpoint="/predict"} 142

# HELP pod_api_predictions_total Total predictions made
# TYPE pod_api_predictions_total counter
pod_api_predictions_total{prediction_type="single"} 142

# HELP pod_api_predicted_pods Predicted pod counts
# TYPE pod_api_predicted_pods counter
pod_api_predicted_pods{pod_type="frontend"} 1420
pod_api_predicted_pods{pod_type="backend"} 568

# HELP pod_api_model_loaded Model loaded status
# TYPE pod_api_model_loaded gauge
pod_api_model_loaded 1
```

**Prometheus Scrape Config:**
```yaml
scrape_configs:
  - job_name: 'pod-forecasting-api'
    static_configs:
      - targets: ['pod-api:5000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

---

### POST /predict

**Single Day Prediction**

**Request:**
```json
{
  "date": "2024-07-15",
  "gmv": 9500000,
  "users": 85000,
  "marketing_cost": 175000
}
```

**Date Formats Supported:**
- ISO: `2024-07-15`
- EU: `15/07/2024`
- US: `07/15/2024`
- Also: `15-07-2024`, `07-15-2024`, `2024/07/15`

**Response:**
```json
{
  "date": "2024-07-15",
  "input": {
    "gmv": 9500000,
    "users": 85000,
    "marketing_cost": 175000
  },
  "predictions": {
    "frontend_pods": 10,
    "backend_pods": 4
  },
  "confidence_intervals": {
    "frontend_pods": [9, 11],
    "backend_pods": [4, 5]
  }
}
```

**Headers:**
```
X-Request-ID: 1728125445123-140234567890
```

**Status Codes:**
- `200` - Success
- `422` - Validation error
- `500` - Prediction error
- `503` - Model not loaded

**cURL Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-07-15",
    "gmv": 9500000,
    "users": 85000,
    "marketing_cost": 175000
  }'
```

**Python Example:**
```python
import requests

response = requests.post('http://localhost:5000/predict', json={
    'date': '2024-07-15',
    'gmv': 9500000,
    'users': 85000,
    'marketing_cost': 175000
})

data = response.json()
print(f"Frontend: {data['predictions']['frontend_pods']} pods")
print(f"Backend: {data['predictions']['backend_pods']} pods")
```

**Node.js Example:**
```javascript
const axios = require('axios');

const response = await axios.post('http://pod-api:5000/predict', {
  date: '2024-07-15',
  gmv: 9500000,
  users: 85000,
  marketing_cost: 175000
});

console.log(response.data.predictions);
// { frontend_pods: 10, backend_pods: 4 }
```

---

### POST /forecast/batch

**Multiple Day Predictions**

**Request:**
```json
{
  "predictions": [
    {
      "date": "2024-07-15",
      "gmv": 9500000,
      "users": 85000,
      "marketing_cost": 175000
    },
    {
      "date": "2024-07-16",
      "gmv": 9800000,
      "users": 86000,
      "marketing_cost": 180000
    }
  ]
}
```

**Response:**
```json
{
  "count": 2,
  "predictions": [
    {
      "date": "2024-07-15",
      "predictions": {
        "frontend_pods": 10,
        "backend_pods": 4
      },
      "confidence_intervals": {
        "frontend_pods": [9, 11],
        "backend_pods": [4, 5]
      }
    },
    {
      "date": "2024-07-16",
      "predictions": {
        "frontend_pods": 10,
        "backend_pods": 4
      },
      "confidence_intervals": {
        "frontend_pods": [9, 11],
        "backend_pods": [4, 5]
      }
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/forecast/batch \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"date": "2024-07-15", "gmv": 9500000, "users": 85000, "marketing_cost": 175000},
      {"date": "2024-07-16", "gmv": 9800000, "users": 86000, "marketing_cost": 180000}
    ]
  }'
```

---

### GET /model-info

**Model Metadata**

**Response:**
```json
{
  "model_type": "RandomForestRegressor",
  "features": [
    "gmv",
    "users",
    "marketing_cost",
    "gmv_per_user",
    "marketing_eff",
    "day_of_week",
    "is_weekend",
    "month",
    "gmv_millions",
    "users_thousands"
  ],
  "n_features": 10,
  "note": "Uses marketing_cost (not marketing_spend)"
}
```

---

### GET /docs

**Interactive API Documentation**

Swagger UI with:
- All endpoints
- Try-it-out functionality
- Request/response examples
- Schema definitions

**Access:** http://localhost:5000/docs

---

## Production Features

### 1. Request Tracing

Every request gets a unique ID:

**Response Headers:**
```
X-Request-ID: 1728125445123-140234567890
```

**Logs:**
```json
{
  "timestamp": "2024-10-05T10:30:45.123Z",
  "level": "INFO",
  "request_id": "1728125445123-140234567890",
  "message": "Prediction successful: FE=10, BE=4"
}
```

**Usage:**
```bash
# Client includes request ID in bug reports
# DevOps can track request through logs
grep "1728125445123-140234567890" logs/pod_api.log
```

---

### 2. Structured Logging

**Log Format:** JSON

**Log Location:** `logs/pod_api.log`

**Example Entry:**
```json
{
  "timestamp": "2024-10-05T10:30:45.123456",
  "level": "INFO",
  "request_id": "1728125445123-140234567890",
  "method": "POST",
  "endpoint": "/predict",
  "duration_ms": 45.2,
  "status_code": 200,
  "message": "Prediction successful",
  "predictions": {
    "frontend_pods": 10,
    "backend_pods": 4
  }
}
```

**View Logs:**
```bash
# Real-time
tail -f logs/pod_api.log

# Search by request ID
grep "1728125445123-140234567890" logs/pod_api.log

# Errors only
grep '"level":"ERROR"' logs/pod_api.log

# Pretty print
cat logs/pod_api.log | jq '.'
```

---

### 3. Exception Handling

**Custom Exceptions:**
```python
ModelNotLoadedError  -> 503 Service Unavailable
PredictionError      -> 500 Internal Server Error
ValidationError      -> 422 Unprocessable Entity
```

**Error Response:**
```json
{
  "error": "Validation failed",
  "message": "GMV must be positive",
  "request_id": "1728125445123-140234567890",
  "status_code": 422
}
```

**All errors include request ID for debugging!**

---

### 4. Prometheus Metrics

**15+ Metrics Exported:**

**Request Metrics:**
- `pod_api_requests_total` - Total requests by endpoint/status
- `pod_api_request_duration_seconds` - Request latency histogram

**Prediction Metrics:**
- `pod_api_predictions_total` - Total predictions by type
- `pod_api_prediction_duration_seconds` - Prediction latency
- `pod_api_predicted_pods` - Pod count distribution

**Model Metrics:**
- `pod_api_model_loaded` - Model status (1=loaded, 0=not loaded)
- `pod_api_model_load_time_seconds` - Model load duration

**Error Metrics:**
- `pod_api_errors_total` - Errors by type and endpoint

**Business Metrics:**
- `pod_api_predicted_pods_count` - Total pods predicted

---

### 5. Health Checks

**Kubernetes-Ready:**

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 2
```

---

## Configuration

### Environment Variables

```bash
# Model paths
export MODEL_PATH=../models/model.pkl
export SCALER_PATH=../models/scaler.pkl

# API settings
export API_PORT=5000
export LOG_LEVEL=INFO

# CORS (if needed)
export CORS_ORIGINS=https://ounass.com,https://admin.ounass.com
```

### .env File

```bash
# Create .env
cat > .env << 'EOF'
MODEL_PATH=../models/model.pkl
SCALER_PATH=../models/scaler.pkl
API_PORT=5000
LOG_LEVEL=INFO
EOF
```

---

## Deployment

### Local Development

```bash
# With reload
uvicorn main:app --reload --port 5000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4
```

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
```

**Build & Run:**
```bash
docker build -t pod-forecasting-api .
docker run -p 5000:5000 -v $(pwd)/../models:/app/models pod-forecasting-api
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pod-forecasting-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pod-forecasting-api
  template:
    metadata:
      labels:
        app: pod-forecasting-api
    spec:
      containers:
      - name: api
        image: your-registry/pod-forecasting-api:latest
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: /models/model.pkl
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: pod-forecasting-api
spec:
  selector:
    app: pod-forecasting-api
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

---

## Monitoring

### Grafana Dashboard

**Query Examples:**

```promql
# Request rate
rate(pod_api_requests_total[5m])

# Error rate
rate(pod_api_errors_total[5m]) / rate(pod_api_requests_total[5m])

# Latency (p95)
histogram_quantile(0.95, rate(pod_api_request_duration_seconds_bucket[5m]))

# Predictions per hour
increase(pod_api_predictions_total[1h])

# Pod distribution
sum by (pod_type) (pod_api_predicted_pods_count)
```

### Alerts

```yaml
groups:
  - name: pod_forecasting_api
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(pod_api_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Pod Forecasting API"
      
      # Model not loaded
      - alert: ModelNotLoaded
        expr: pod_api_model_loaded == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Pod Forecasting model not loaded"
      
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(pod_api_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in Pod Forecasting API"
```

---

## Testing

### Unit Tests

```python
# test_api.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "date": "2024-07-15",
        "gmv": 9500000,
        "users": 85000,
        "marketing_cost": 175000
    })
    assert response.status_code == 200
    assert "predictions" in response.json()
```

**Run:**
```bash
pytest test_api.py
```

---

## Troubleshooting

### Model Not Loading

**Symptom:** `503 Service Unavailable` on all requests

**Check:**
```bash
# Model files exist?
ls -lh ../models/model.pkl ../models/scaler.pkl

# Retrain if missing
cd ../scripts && python train_model.py
```

### High Latency

**Symptom:** Slow response times

**Check:**
```bash
# View metrics
curl http://localhost:5000/metrics | grep duration

# Check logs
grep "duration_ms" logs/pod_api.log | tail -20
```

### Memory Issues

**Symptom:** API crashes or OOM errors

**Solution:**
```bash
# Check model size
ls -lh ../models/model.pkl

# Increase container memory
# Or reduce model complexity in training
```

---

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Start dev server
uvicorn main:app --reload --port 5000

# Start production
uvicorn main:app --host 0.0.0.0 --port 5000 --workers 4

# Test health
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict -d '{"date":"2024-07-15","gmv":9500000,"users":85000,"marketing_cost":175000}'

# View docs
open http://localhost:5000/docs

# View metrics
curl http://localhost:5000/metrics

# View logs
tail -f logs/pod_api.log
```

---

## Summary

**FastAPI service provides:**
- REST API for pod predictions
- Interactive documentation
- Prometheus metrics
- Health checks
- Request tracing
- Structured logging
- Multi-format date support
- Production-ready error handling

**Ready for Kubernetes deployment with full observability!**