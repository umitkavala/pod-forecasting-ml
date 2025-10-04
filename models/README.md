# Model Artifacts

This directory contains all model-related files for the pod forecasting ML pipeline.

## Contents

- `model.pkl` : Trained Random Forest model for pod forecasting
- `scaler.pkl` : Scaler used for feature normalization
- `model_info.json` : Metadata and training details for the current model

## Usage

These files are automatically generated and updated by the training pipeline (`scripts/train_model.py`).

- To retrain the model, run:
  ```bash
  uv run scripts/train_model.py
  ```
- To use the model for predictions, see the API integration in `api/predictor.py`.

## Versioning

- Model version and training details are tracked in `model_info.json`.
- For reproducibility, always keep this directory under version control.

## Notes

- Do not manually edit model files.
- For more details, see the main project [README.md](../README.md).
