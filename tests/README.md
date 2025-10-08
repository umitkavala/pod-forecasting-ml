# Tests Directory

This folder contains unit tests for the pod forecasting ML pipeline.

## Contents
- `test_api.py`              : Tests for API endpoints (health, predict, metrics) using TestClient
- `test_predictor.py`        : Unit tests for `api.predictor.PodPredictor` (feature engineering, prediction)
- `test_fetch_from_sheets.py`: Tests for Google Sheets data fetching and validation (mocks for API errors/retries)
- `test_clean_data.py`       : Tests for data cleaning, missing values, and outlier handling
- `test_train_model.py`      : Tests for model training, feature engineering, and evaluation

## Running Tests

To run all tests (from project root, using the project's virtualenv):
```bash
# activate your venv, then
./.venv/bin/python -m unittest discover -v tests
```

To run a specific test file:
```bash
./.venv/bin/python -m unittest tests.test_clean_data -v
```

To run a single test method in a file:
```bash
./.venv/bin/python -m unittest tests.test_clean_data.TestCleanData.test_check_duplicates -v
```

## Notes
- All tests use Python's `unittest` framework
- Ensure dependencies are installed (see `requirements.txt` and `./.venv`) before running tests
- Tests are designed to run in isolation and do not require external data
- For more details, see the main project [README.md](../README.md)

## New tests (recent additions)
The following tests were recently added to improve coverage for API, predictor logic, and Google Sheets integration:

- `test_api.py` - TestAPI
	- Tests the FastAPI endpoints (`/health`, `/predict`, `/metrics`) using `TestClient` and a `DummyPredictor`.
	- Run: `./.venv/bin/python -m unittest tests.test_api -v`

- `test_predictor.py` - TestPodPredictor
	- Unit tests for `api.predictor.PodPredictor` with `joblib.load` patched to return a fake model and scaler. Tests `engineer_features`, `predict`, and `get_model_info`.
	- Run: `./.venv/bin/python -m unittest tests.test_predictor -v`

- `test_fetch_from_sheets.py` - TestFetchFromSheets
	- Expanded mocks for Google Sheets API: success, empty sheet, 404/403 errors, and retry behavior.
	- Run: `./.venv/bin/python -m unittest tests.test_fetch_from_sheets -v`

These tests are part of the standard suite and will run with the main discovery command above.
