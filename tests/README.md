# Tests Directory

This folder contains unit tests for the pod forecasting ML pipeline.

## Contents
- `test_fetch_from_sheets.py` : Tests for Google Sheets data fetching and validation
- `test_clean_data.py`        : Tests for data cleaning, missing values, and outlier handling
- `test_train_model.py`       : Tests for model training, feature engineering, and evaluation

## Running Tests

To run all tests:
```bash
uv run -m unittest discover tests
```

To run a specific test file:
```bash
uv run -m unittest tests/test_clean_data.py
```

## Notes
- All tests use Python's `unittest` framework
- Ensure dependencies are installed before running tests
- Tests are designed to run in isolation and do not require external data
- For more details, see the main project [README.md](../README.md)
