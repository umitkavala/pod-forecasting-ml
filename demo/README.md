# API Integration Examples

Working examples demonstrating how to integrate with the Pod Forecasting API.

---

## Overview

This folder contains runnable examples showing how to call the Pod Forecasting API:

- **Python** (`demo.py`) - Using `requests` library
- **Node.js** (`nodejs_integration.js`) - Using `axios` library

Both examples demonstrate:
- API authentication with API keys
- Making prediction requests
- Handling responses and errors

---

## Prerequisites

### 1. API Must Be Running

```bash
cd api
uvicorn main:app --port 5000
```

### 2. API Key Configured

```bash
export API_KEY_NODE_SERVICE="your-api-key-here"
```

---

## Python Example

### Setup

```bash
# Install dependencies
pip install requests
```

### Run

```bash
python demo.py
```

### Expected Output

```
=============================================================
Pod Forecasting API - Python Integration Demo
=============================================================

Testing single prediction...
Prediction successful!
  Date: 2024-07-15
  Predicted pods: FE=10, BE=4
  Confidence: FE=[9-11], BE=[4-5]

Testing batch predictions...
Batch prediction successful!
  Processed 3 predictions

=============================================================
All tests passed!
=============================================================
```

### Code Example

```python
import requests
import os

# Configuration
API_URL = "http://localhost:5000"
API_KEY = os.getenv("API_KEY_NODE_SERVICE")

# Headers with API key
headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# Make prediction request
response = requests.post(
    f"{API_URL}/predict",
    headers=headers,
    json={
        "date": "2024-07-15",
        "gmv": 9500000,
        "users": 85000,
        "marketing_cost": 175000
    }
)

# Check response
if response.status_code == 200:
    data = response.json()
    print(f"Frontend pods: {data['predictions']['frontend_pods']}")
    print(f"Backend pods: {data['predictions']['backend_pods']}")
else:
    print(f"Error: {response.json()}")
```

---

## Node.js Example

### Setup

```bash
# Install dependencies
npm install
```

### Run

```bash
node nodejs_integration.js
```

### Expected Output

```
=============================================================
Pod Forecasting API - Node.js Integration Demo
=============================================================

Testing single prediction...
Prediction successful!
  Date: 2024-07-15
  Predicted pods: FE=10, BE=4
  Confidence: FE=[9-11], BE=[4-5]

Testing batch predictions...
Batch prediction successful!
  Processed 3 predictions

=============================================================
All tests passed!
=============================================================
```

### Code Example

```javascript
const axios = require('axios');

// Configuration
const API_URL = 'http://localhost:5000';
const API_KEY = process.env.API_KEY_NODE_SERVICE;

// Make prediction request
const response = await axios.post(
    `${API_URL}/predict`,
    {
        date: '2024-07-15',
        gmv: 9500000,
        users: 85000,
        marketing_cost: 175000
    },
    {
        headers: {
            'X-API-Key': API_KEY,
            'Content-Type': 'application/json'
        }
    }
);

// Check response
console.log('Frontend pods:', response.data.predictions.frontend_pods);
console.log('Backend pods:', response.data.predictions.backend_pods);
```

---

## API Endpoints

### Single Prediction

**Endpoint:** `POST /predict`

**Request:**
```json
{
    "date": "2024-07-15",
    "gmv": 9500000,
    "users": 85000,
    "marketing_cost": 175000
}
```

**Response:**
```json
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
}
```

### Batch Predictions

**Endpoint:** `POST /forecast/batch`

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
            "gmv": 10000000,
            "users": 90000,
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
                "frontend_pods": 11,
                "backend_pods": 4
            },
            "confidence_intervals": {
                "frontend_pods": [10, 12],
                "backend_pods": [4, 5]
            }
        }
    ]
}
```

---

## Error Handling

### Missing API Key

**Response:**
```json
{
    "error": "Missing API key",
    "request_id": "1728125445456-140234567891"
}
```

**Status Code:** `401 Unauthorized`

### Invalid API Key

**Response:**
```json
{
    "error": "Invalid API key",
    "request_id": "1728125445456-140234567891"
}
```

**Status Code:** `401 Unauthorized`

### Model Not Loaded

**Response:**
```json
{
    "error": "Model not loaded",
    "request_id": "1728125445456-140234567891"
}
```

**Status Code:** `503 Service Unavailable`

---

## Troubleshooting

### Connection Refused

**Problem:** Can't connect to API

**Solution:**
- Make sure API is running: `uvicorn api.main:app --port 5000`
- Check the port (should be 5000)

### 401 Unauthorized

**Problem:** Invalid or missing API key

**Solution:**
- Check API key is set: `echo $API_KEY_NODE_SERVICE`
- Make sure key matches the one configured in the API

### 503 Service Unavailable

**Problem:** Model not loaded

**Solution:**
- Check API logs for errors
- Make sure model files exist in `models/` directory
- Restart the API

### Import/Module Errors

**Python:**
```bash
pip install requests
```

**Node.js:**
```bash
npm install
```

---

## Files

- `demo.py` - Python integration example
- `nodejs_integration.js` - Node.js integration example
- `package.json` - Node.js dependencies
- `README.md` - This file

---

## Next Steps

- **API Documentation:** See `../api/README.md` for complete API reference
- **Authentication Details:** See `../docs/API_AUTHENTICATION_GUIDE.md`
- **Setup Google Sheets:** See `../docs/SERVICE_ACCOUNT_SETUP_GUIDE.md`

---
