# Scripts Directory - Data Pipeline

Machine learning data pipeline for pod forecasting. Handles data ingestion from Google Sheets, cleaning, validation, and model training.

---

## Overview

**Pipeline Flow:**
```
Google Sheets -> Fetch -> Clean -> Train -> Deploy
```

**Scripts:**
1. `fetch_from_sheets.py` - OAuth-based data ingestion
2. `fetch_from_sheets_SERVICE_ACCOUNT.py` - Service account for cronjobs
3. `clean_data.py` - Data cleaning and validation
4. `train_model.py` - Model training and evaluation

---

## Scripts

### 1. fetch_from_sheets.py (OAuth)

**Purpose:** Fetch data from Google Sheets using OAuth user consent

**When to use:**
- Manual runs
- Development
- First-time setup

**Authentication:**
- Requires `credentials.json`
- Opens browser for consent
- Saves token to `token.pickle`

**Usage:**
```bash
python fetch_from_sheets.py
```

**Output:**
- `../data/historical_raw.csv`
- `../data/budget_raw.csv`
- `logs/data_ingestion.log`

**Setup:**
1. Download `credentials.json` from Google Cloud Console
2. Place in `scripts/` directory
3. Run script (browser will open)
4. Click "Allow"

**Not suitable for cronjobs!**

---

### 2. fetch_from_sheets_SERVICE_ACCOUNT.py (Service Account)

**Purpose:** Fetch data using service account (no user interaction)

**When to use:**
- Automated cronjobs
- Production environments
- CI/CD pipelines

**Authentication:**
- Requires `service-account-key.json`
- No browser needed
- No token expiry

**Usage:**
```bash
export SPREADSHEET_ID="your_spreadsheet_id"
export SERVICE_ACCOUNT_FILE="../service-account-key.json"
python fetch_from_sheets_SERVICE_ACCOUNT.py
```

**Output:**
- `../data/historical_raw.csv`
- `../data/budget_raw.csv`
- `logs/data_ingestion.log`

**Setup:**
1. Create service account in Google Cloud Console
2. Download JSON key
3. Share Google Sheet with service account email
4. Set environment variables
5. Run script

**Suitable for cronjobs!**

---

### 3. clean_data.py

**Purpose:** Clean and validate fetched data

**Features:**
- Remove commas from numbers
- Parse dates (DD/MM/YYYY -> YYYY-MM-DD)
- Validate data quality
- Detect outliers
- Handle missing values
- Remove duplicates

**Usage:**
```bash
python clean_data.py
```

**Input:**
- `../data/historical_raw.csv`
- `../data/budget_raw.csv`

**Output:**
- `../data/training_clean.csv`
- `../data/budget_clean.csv`
- `logs/data_cleaning.log`

**Quality Checks:**
```
1. Date parsing (DD/MM/YYYY format)
2. Comma removal ("18,953,653" -> 18953653.0)
3. Type conversion (string -> numeric)
4. Missing value detection
5. Duplicate detection
6. Outlier detection
7. Data validation
```

**Example Output:**
```
======================================================================
  STARTING DATA CLEANING
======================================================================

STEP 1: Loading Historical Data
  Loaded 30 rows from ../data/historical_raw.csv

STEP 2: Cleaning Historical Data

1. Parsing dates...
  Dates parsed successfully (dayfirst=True for DD/MM/YYYY)
  Date range: 2024-07-15 to 2024-08-13

2. Converting numeric columns...
  Cleaning gmv - removing commas and converting to numeric
  gmv converted to numeric successfully
  Cleaning users - removing commas and converting to numeric
  users converted to numeric successfully
  Cleaning marketing_cost - removing commas and converting to numeric
  marketing_cost converted to numeric successfully

3. Handling missing values...
  No missing values to handle

4. Removing duplicates...
  No duplicates to remove

5. Sorting by date...
  Data sorted by date

DATA QUALITY CHECKS: Historical
======================================================================
  All quality checks passed for Historical
======================================================================

  DATA CLEANING COMPLETE
======================================================================

Summary:
   Training data:  30 rows, 6 columns
   Budget data:    334 rows, 4 columns
   Duration:       1.23 seconds

Files saved:
   - ../data/training_clean.csv
   - ../data/budget_clean.csv

Next: python train_model.py
```

---

### 4. train_model.py

**Purpose:** Train Random Forest model on cleaned data

**Features:**
- Feature engineering (10 features)
- Train/test split
- Model training
- Performance evaluation
- Overfitting detection
- Model saving with metadata

**Usage:**
```bash
python train_model.py
```

**Input:**
- `../data/training_clean.csv`

**Output:**
- `../models/model.pkl`
- `../models/scaler.pkl`
- `../models/model_info.json`
- `logs/model_training.log`
- `logs/training_report.json`

**Model Features:**
```python
[
    'gmv',              # Raw GMV
    'users',            # Raw users
    'marketing_cost',   # Raw marketing cost
    'gmv_per_user',     # GMV / users
    'marketing_eff',    # GMV / marketing_cost
    'day_of_week',      # 0-6 (Monday-Sunday)
    'is_weekend',       # 0 or 1
    'month',            # 1-12
    'gmv_millions',     # GMV / 1M
    'users_thousands'   # Users / 1K
]
```

**Example Output:**
```
======================================================================
  STARTING MODEL TRAINING PIPELINE
======================================================================

STEP 1: Loading training data...
  Loaded 30 training samples

STEP 2: Engineering features...
  Created gmv_per_user
  Created marketing_eff
  Created temporal features
  Created scaled features
  Feature matrix: (30, 10)
  Target matrix: (30, 2)

STEP 3: Scaling features...
  Features scaled

STEP 4: Splitting data (75% train, 25% test)...
  Training set: 22 samples
  Test set: 8 samples

STEP 5: Training model...
  Model trained in 0.15 seconds

STEP 6: Evaluating on training set...

  Frontend Pods (Training Set):
    MAE:  0.12 pods
    RMSE: 0.18 pods
    R²:   0.998

  Backend Pods (Training Set):
    MAE:  0.08 pods
    RMSE: 0.12 pods
    R²:   0.999

STEP 7: Evaluating on test set (unseen data)...

  Frontend Pods (Test Set):
    MAE:  1.25 pods
    RMSE: 1.87 pods
    R²:   0.94

  Backend Pods (Test Set):
    MAE:  0.75 pods
    RMSE: 1.02 pods
    R²:   0.96

STEP 8: Checking for overfitting...
  Train-Test MAE gap:
    Frontend: 1.13 pods
    Backend:  0.67 pods
  Reasonable train-test gap - model generalizes well

STEP 9: Validating performance...
  Model performance meets all thresholds

STEP 10: Retraining on full dataset for deployment...
  Model trained in 0.18 seconds

STEP 11: Saving model artifacts...
  Saved model to ../models/model.pkl
  Saved scaler to ../models/scaler.pkl
  Saved metadata to ../models/model_info.json
  Saved training report to logs/training_report.json

  MODEL TRAINING COMPLETE
======================================================================

Final Performance (Test Set):
   Frontend MAE: 1.25 pods
   Backend MAE:  0.75 pods
   Frontend R²:  0.94
   Backend R²:   0.96

Performance:
   Training time: 2.45s
   Prediction latency: 35.2ms/sample

Model artifacts:
   - ../models/model.pkl
   - ../models/scaler.pkl
   - ../models/model_info.json
   - logs/training_report.json

Next: cd ../api && uvicorn main:app --port 5000
```

---

## Pipeline Execution

### Manual Run (Step-by-Step)

```bash
cd scripts

# Step 1: Fetch data
python fetch_from_sheets_SERVICE_ACCOUNT.py

# Step 2: Clean data
python clean_data.py

# Step 3: Train model
python train_model.py
```

### Automated Run (Pipeline Script)

Create `run_pipeline.sh`:
```bash
#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Starting Pipeline: $(date) ==="

# 1. Fetch
echo "1. Fetching data..."
python fetch_from_sheets_SERVICE_ACCOUNT.py || exit 1

# 2. Clean
echo "2. Cleaning data..."
python clean_data.py || exit 1

# 3. Train
echo "3. Training model..."
python train_model.py || exit 1

echo "=== Pipeline Complete: $(date) ==="
```

**Run:**
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## Configuration

### Environment Variables

```bash
# Google Sheets
export SPREADSHEET_ID="1ABC123xyz456DEF789"
export SERVICE_ACCOUNT_FILE="../service-account-key.json"

# Or use .env file
cat > .env << 'EOF'
SPREADSHEET_ID=1ABC123xyz456DEF789
SERVICE_ACCOUNT_FILE=../service-account-key.json
EOF
```

### Script Configuration

**fetch_from_sheets_SERVICE_ACCOUNT.py:**
```python
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', 'YOUR_SPREADSHEET_ID')
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE', 'service-account-key.json')
```

**clean_data.py:**
```python
# Data quality thresholds
MIN_GMV = 0
MAX_GMV = 100_000_000
MIN_USERS = 0
MAX_USERS = 1_000_000
MIN_MARKETING_COST = 0
MAX_MARKETING_COST = 10_000_000
```

**train_model.py:**
```python
# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 20,
    'max_depth': 4,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'random_state': 42
}

# Training configuration
TEST_SIZE = 0.25
RANDOM_STATE = 42
```

---

## Logs

All scripts generate detailed logs in `logs/` directory:

```
logs/
├── data_ingestion.log     # Google Sheets fetch logs
├── data_cleaning.log      # Data quality logs
├── model_training.log     # Training process logs
└── training_report.json   # Training metrics (JSON)
```

**View logs:**
```bash
# Real-time
tail -f logs/data_ingestion.log

# Search errors
grep "ERROR" logs/*.log

# Last run summary
tail -50 logs/model_training.log
```

---

## Cronjob Setup

### Daily Pipeline

```bash
# Edit crontab
crontab -e

# Run daily at 2 AM
0 2 * * * cd /path/to/pod-forecasting/scripts && ./run_pipeline.sh >> /var/log/pod-forecasting.log 2>&1
```

### Hourly Data Refresh

```bash
# Fetch new data hourly
0 * * * * cd /path/to/pod-forecasting/scripts && python fetch_from_sheets_SERVICE_ACCOUNT.py >> /var/log/pod-fetch.log 2>&1

# Clean and retrain every 6 hours
0 */6 * * * cd /path/to/pod-forecasting/scripts && python clean_data.py && python train_model.py >> /var/log/pod-train.log 2>&1
```

### With Error Alerts

```bash
#!/bin/bash
cd /path/to/pod-forecasting/scripts

# Run pipeline
./run_pipeline.sh

# Alert on failure
if [ $? -ne 0 ]; then
    echo "Pipeline failed at $(date)" | mail -s "Pod Forecasting Alert" admin@ounass.com
fi
```

---

## Error Handling

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (check logs) |

**Example:**
```bash
python fetch_from_sheets_SERVICE_ACCOUNT.py
echo $?  # 0 = success, 1 = failure
```

### Common Errors

**1. Service Account Authentication Failed**
```
ERROR: Service account file not found
```
**Solution:** Check file path, download from Google Cloud Console

**2. Permission Denied (403)**
```
ERROR: Service account lacks permission
```
**Solution:** Share Google Sheet with service account email

**3. Data Validation Error**
```
ERROR: Historical missing columns: ['fe_pods']
```
**Solution:** Check Google Sheets structure, verify column names

**4. Training Failed**
```
ERROR: Insufficient training data: 5 samples (minimum: 10)
```
**Solution:** Add more historical data to Google Sheets

---

## Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Key packages:**
```
pandas>=2.0.3
numpy>=1.24.3
scikit-learn>=1.3.0
google-auth>=2.23.0
google-auth-oauthlib>=1.1.0
google-api-python-client>=2.100.0
joblib>=1.3.2
python-dotenv>=1.0.0
```

---

## Data Flow

```
┌─────────────────────────┐
│   Google Sheets         │
│   - Historical sheet    │
│   - Budget sheet        │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ fetch_from_sheets.py    │
│ - Authenticate          │
│ - Fetch data            │
│ - Validate structure    │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ ../data/                │
│ - historical_raw.csv    │
│ - budget_raw.csv        │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ clean_data.py           │
│ - Remove commas         │
│ - Parse dates           │
│ - Validate quality      │
│ - Handle missing        │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ ../data/                │
│ - training_clean.csv    │
│ - budget_clean.csv      │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ train_model.py          │
│ - Engineer features     │
│ - Train Random Forest   │
│ - Evaluate performance  │
│ - Save model            │
└───────────┬─────────────┘
            │
            v
┌─────────────────────────┐
│ ../models/              │
│ - model.pkl             │
│ - scaler.pkl            │
│ - model_info.json       │
└─────────────────────────┘
```

---

## Troubleshooting

### Check Pipeline Status

```bash
# Check if all files created
ls -lh ../data/*.csv
ls -lh ../models/*.pkl

# Check logs for errors
grep -i "error" logs/*.log

# Verify data quality
python -c "import pandas as pd; df = pd.read_csv('../data/training_clean.csv'); print(df.info())"
```

### Validate Model

```bash
# Check model exists
ls -lh ../models/model.pkl

# Load and test
python -c "
import joblib
model = joblib.load('../models/model.pkl')
print('Model loaded:', type(model))
"
```

### Test End-to-End

```bash
# Run full pipeline
./run_pipeline.sh

# Should complete without errors
echo $?  # Should output: 0
```

---

## Monitoring

### Pipeline Health Checks

```bash
# Check last successful run
ls -lt ../data/*.csv | head -1

# Check model age
find ../models -name "model.pkl" -mtime +7  # Models older than 7 days

# Count training samples
wc -l ../data/training_clean.csv
```

### Automated Monitoring

```bash
#!/bin/bash
# monitor_pipeline.sh

# Check data freshness
if [ ! -f "../data/training_clean.csv" ]; then
    echo "ERROR: No training data found"
    exit 1
fi

# Check model exists
if [ ! -f "../models/model.pkl" ]; then
    echo "ERROR: No model found"
    exit 1
fi

# Check data age
AGE=$(find ../data/training_clean.csv -mtime +1)
if [ -n "$AGE" ]; then
    echo "WARNING: Data is stale (>24 hours old)"
fi

echo "Pipeline health: OK"
```

---

## Best Practices

1. **Run pipeline daily** to keep model fresh
2. **Monitor logs** for errors and warnings
3. **Validate data** before training
4. **Version control** model artifacts
5. **Test changes** on sample data first
6. **Backup data** before major changes
7. **Use service account** for automation
8. **Set up alerts** for failures

---

## Quick Reference

```bash
# OAuth fetch (manual)
python fetch_from_sheets.py

# Service account fetch (cronjob)
python fetch_from_sheets_SERVICE_ACCOUNT.py

# Clean data
python clean_data.py

# Train model
python train_model.py

# Full pipeline
./run_pipeline.sh

# View logs
tail -f logs/model_training.log

# Check status
ls -lh ../models/*.pkl ../data/*.csv
```

---

## Files

| File | Purpose | Output |
|------|---------|--------|
| `fetch_from_sheets.py` | OAuth data fetch | raw CSVs |
| `fetch_from_sheets_SERVICE_ACCOUNT.py` | Service account fetch | raw CSVs |
| `clean_data.py` | Data cleaning | clean CSVs |
| `train_model.py` | Model training | model files |
| `run_pipeline.sh` | Full pipeline | all outputs |

---

## Summary

**Scripts directory contains the complete ML pipeline:**

1. Fetch data from Google Sheets
2. Clean and validate
3. Train Random Forest model
4. Save for API deployment

**All steps are logged, validated, and production-ready!**