# Pod Forecasting System

ML-powered Kubernetes pod forecasting for Ounass e-commerce. Predicts optimal frontend and backend pod counts based on GMV, users, and marketing spend.

---

## Quick Start

```bash
# 1. Setup
git clone <repo> && cd pod-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure (see scripts/README.md for Google Cloud setup)
export SPREADSHEET_ID="your_id"
export SERVICE_ACCOUNT_FILE="service-account-key.json"

# 3. Run pipeline
cd scripts
python fetch_from_sheets_SERVICE_ACCOUNT.py
python clean_data.py
python train_model.py

# 4. Start API
cd ../api && uvicorn main:app --port 5000

# 5. Test
curl -X POST http://localhost:5000/predict \
  -d '{"date":"2024-07-15","gmv":9500000,"users":85000,"marketing_cost":175000}'
```

---

## Project Structure

```
pod-forecasting/
├── api/                    # FastAPI service -> See api/README.md
├── scripts/                # Data pipeline -> See scripts/README.md
├── data/                   # Data files -> See data/README.md
├── models/                 # Trained models
└── demo/                   # Demo apps (python & node)
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `POST /predict` | Single prediction |
| `POST /forecast/batch` | Batch predictions |
| `GET /docs` | Interactive docs at http://localhost:5000/docs |

Full API documentation: [api/README.md](api/README.md)

---

## Integration Example

```javascript
// Node.js
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

## Key Features

- Random Forest model (MAE < 2 pods, R² > 0.90)
- FastAPI with Prometheus metrics
- Google Sheets integration
- Multi-format date support (DD/MM/YYYY, YYYY-MM-DD, MM/DD/YYYY)
- Automated cronjobs with service account

---

## Deployment

```bash
# Docker
docker-compose build
docker-compose up -d 
  
```

Deployment details: [api/README.md#deployment](api/README.md)

---

## Monitoring

- **API Logs:** `api/logs/pod_api.log`
- **Pipeline Logs:** `scripts/logs/*.log`
- **Metrics:** http://localhost:5000/metrics
- **Health:** http://localhost:5000/health

Monitoring setup: [api/README.md#monitoring](api/README.md)

---

## Documentation

| Document | Description |
|----------|-------------|
| **[api/README.md](api/README.md)** | Complete API docs, endpoints, deployment, monitoring |
| **[scripts/README.md](scripts/README.md)** | Pipeline, cronjobs, automation, troubleshooting |
| **[data/README.md](data/README.md)** | Data formats, quality checks, validation |
| **[docs/outputs/](docs/outputs/)** | Google Cloud setup, date formats, guides |

---

## Tech Stack

Python, FastAPI, scikit-learn, Google Sheets API, Prometheus, Docker, Kubernetes

---

**Built for Ounass Head of Engineering interview**

Production-ready system. See individual README files for details.
