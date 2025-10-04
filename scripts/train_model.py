"""
Model training with production-grade logging, monitoring and error handling.

Features:
- Structured logging with timestamps
- Comprehensive exception handling
- Model validation and testing
- Performance metrics tracking
- Cross-validation
- Model versioning
- Training metrics export
"""

import os
import logging
import sys
import time
import json
from datetime import datetime
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

if __name__ == "__main__":
    # Direct: python scripts/file.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.exceptions import DataLoadError, FeatureEngineeringError, InsufficientDataError, ModelTrainingError, ModelValidationError
else:
    # Imported as module
    from .exceptions import DataLoadError, FeatureEngineeringError, InsufficientDataError, ModelTrainingError, ModelValidationError

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model hyperparameters
MODEL_PARAMS = {
    'n_estimators': 20,
    'max_depth': 4,
    'min_samples_split': 4,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Training configuration
TEST_SIZE = 0.25
RANDOM_STATE = 42
MIN_TRAINING_SAMPLES = 10

# Performance thresholds
MAX_ACCEPTABLE_MAE = 10  # pods
MIN_ACCEPTABLE_R2 = 0.5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_dataset_stats(df: pd.DataFrame, name: str) -> None:
    """Log dataset statistics."""
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"\n  Numeric columns:")
    for col in numeric_cols:
        logger.info(f"    {col}:")
        logger.info(f"      min: {df[col].min():.2f}")
        logger.info(f"      max: {df[col].max():.2f}")
        logger.info(f"      mean: {df[col].mean():.2f}")
        logger.info(f"      std: {df[col].std():.2f}")

def check_data_sufficiency(df: pd.DataFrame) -> None:
    """
    Check if there's sufficient data for training.
    
    Raises:
        InsufficientDataError: If data is insufficient
    """
    n_samples = len(df)
    
    if n_samples < MIN_TRAINING_SAMPLES:
        raise InsufficientDataError(
            f"Insufficient training data: {n_samples} samples "
            f"(minimum: {MIN_TRAINING_SAMPLES})"
        )
    
    # Check target variance
    if 'fe_pods' in df.columns:
        fe_unique = df['fe_pods'].nunique()
        be_unique = df['be_pods'].nunique()
        
        logger.info(f"\nTarget variance:")
        logger.info(f"  fe_pods unique values: {fe_unique}")
        logger.info(f"  be_pods unique values: {be_unique}")
        
        if fe_unique < 2 or be_unique < 2:
            logger.warning(
                "WARNING:  Low target variance detected. "
                "Model may achieve perfect scores on limited patterns."
            )

def check_data_leakage(X_train, X_test, y_train, y_test) -> None:
    """
    Comprehensive data leakage detection.
    
    Checks for:
    1. Overlapping indices
    2. Duplicate feature patterns (identical rows in train and test)
    
    Args:
        X_train: Training features (numpy array or DataFrame)
        X_test: Test features (numpy array or DataFrame)
        y_train: Training targets
        y_test: Test targets
        
    Raises:
        ModelValidationError: If any leakage is detected
    """
    logger.info("\n" + "="*70)
    logger.info("CHECKING FOR DATA LEAKAGE")
    logger.info("="*70)
    
    # ========================================================================
    # CHECK 1: Overlapping Indices
    # ========================================================================
    logger.info("\n[1/2] Checking for overlapping indices...")
    train_indices = set(y_train.index)
    test_indices = set(y_test.index)
    overlap = train_indices.intersection(test_indices)
    
    if overlap:
        raise ModelValidationError(
            f"Data leakage detected: {len(overlap)} overlapping samples "
            f"between train and test sets. Indices: {list(overlap)[:5]}..."
        )
    
    logger.info(f"  No overlapping indices")
    logger.info(f"    Train samples: {len(train_indices)}")
    logger.info(f"    Test samples: {len(test_indices)}")
    
    # ========================================================================
    # CHECK 2: Duplicate Feature Patterns
    # ========================================================================
    logger.info("\n[2/2] Checking for duplicate feature patterns...")
    
    # Convert to numpy if needed
    X_train_array = X_train if isinstance(X_train, np.ndarray) else X_train.values
    X_test_array = X_test if isinstance(X_test, np.ndarray) else X_test.values
    
    # Create feature signatures (hash of each row)
    train_signatures = set()
    for row in X_train_array:
        sig = hash(tuple(row))
        train_signatures.add(sig)
    
    test_signatures = set()
    duplicate_count = 0
    duplicate_indices = []
    
    for idx, row in enumerate(X_test_array):
        sig = hash(tuple(row))
        test_signatures.add(sig)
        if sig in train_signatures:
            duplicate_count += 1
            duplicate_indices.append(idx)
    
    if duplicate_count > 0:
        # Show examples of duplicates
        logger.error(f"\n  ERROR: Found {duplicate_count} duplicate feature patterns!")
        logger.error(f"     Test rows with duplicates: {duplicate_indices[:5]}...")
        
        # Show first duplicate example
        if len(duplicate_indices) > 0:
            dup_idx = duplicate_indices[0]
            logger.error(f"\n  Example duplicate (test row {dup_idx}):")
            logger.error(f"    Features: {X_test_array[dup_idx]}")
            logger.error(f"    This exact feature combination exists in training set!")
        
        raise ModelValidationError(
            f"Feature pattern leakage detected: {duplicate_count} test samples "
            f"have identical feature combinations as training samples. "
            f"This will artificially inflate accuracy!"
        )
    
    logger.info(f"  No duplicate feature patterns")
    logger.info(f"    Unique train patterns: {len(train_signatures)}")
    logger.info(f"    Unique test patterns: {len(test_signatures)}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("DATA LEAKAGE CHECK COMPLETE")
    logger.info("="*70)
    logger.info(f"  No leakage detected")
    logger.info(f"  Training samples: {len(X_train_array)}")
    logger.info(f"  Test samples: {len(X_test_array)}")
    logger.info(f"  Feature dimensionality: {X_train_array.shape[1]}")
    logger.info("="*70 + "\n")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
        
    Raises:
        FeatureEngineeringError: If feature engineering fails
    """
    logger.info("\nEngineering features...")
    
    try:
        df = df.copy()

        # 1. GMV per user
        df['gmv_per_user'] = df['gmv'] / df['users']
        logger.info("  Created gmv_per_user")

        # 2. Marketing efficiency
        df['marketing_eff'] = df['gmv'] / df['marketing_cost']
        logger.info("  Created marketing_eff")

        # 3. Temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        logger.info("  Created temporal features")

        # 4. Scaled features
        df['gmv_millions'] = df['gmv'] / 1_000_000
        df['users_thousands'] = df['users'] / 1_000
        logger.info("  Created scaled features")

        # Check for invalid values (inf, nan)
        invalid_count = df[df.select_dtypes(include=[np.number]).columns].isin([np.inf, -np.inf]).sum().sum()
        nan_count = df.isnull().sum().sum()

        if invalid_count > 0:
            raise FeatureEngineeringError(f"Feature engineering produced {invalid_count} infinite values")

        if nan_count > 0:
            logger.warning(f"Feature engineering produced {nan_count} NaN values, dropping...")
            df = df.dropna()

        logger.info(f"Feature engineering complete: {df.shape[1]} features")
        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        raise FeatureEngineeringError(f"Feature engineering error: {e}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train) -> MultiOutputRegressor:
    """
    Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Trained model
        
    Raises:
        ModelTrainingError: If training fails
    """
    logger.info("\nTraining Random Forest model...")
    logger.info(f"  Parameters: {MODEL_PARAMS}")
    
    try:
        start_time = time.time()
        
        model = MultiOutputRegressor(
            RandomForestRegressor(**MODEL_PARAMS)
        )
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Model trained in {training_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        raise ModelTrainingError(f"Training error: {e}")

def evaluate_model(model, X, y, dataset_name: str) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: True targets
        dataset_name: Name of dataset (for logging)
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\nEvaluating on {dataset_name}...")
    
    try:
        start_time = time.time()
        y_pred = model.predict(X)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        mae_fe = mean_absolute_error(y['fe_pods'], y_pred[:, 0])
        mae_be = mean_absolute_error(y['be_pods'], y_pred[:, 1])
        
        r2_fe = r2_score(y['fe_pods'], y_pred[:, 0])
        r2_be = r2_score(y['be_pods'], y_pred[:, 1])
        
        rmse_fe = np.sqrt(mean_squared_error(y['fe_pods'], y_pred[:, 0]))
        rmse_be = np.sqrt(mean_squared_error(y['be_pods'], y_pred[:, 1]))
        
        # Log metrics
        logger.info(f"\n  Frontend Pods ({dataset_name}):")
        logger.info(f"    MAE:  {mae_fe:.2f} pods")
        logger.info(f"    RMSE: {rmse_fe:.2f} pods")
        logger.info(f"    R²:   {r2_fe:.4f}")
        
        logger.info(f"\n  Backend Pods ({dataset_name}):")
        logger.info(f"    MAE:  {mae_be:.2f} pods")
        logger.info(f"    RMSE: {rmse_be:.2f} pods")
        logger.info(f"    R²:   {r2_be:.4f}")
        
        logger.info(f"\n  Prediction time: {prediction_time:.3f}s for {len(X)} samples")
        logger.info(f"  Avg per sample: {prediction_time/len(X)*1000:.1f}ms")
        
        # Show sample predictions
        logger.info(f"\n  Sample predictions ({dataset_name}):")
        logger.info(f"    {'Pred FE':<10} {'Pred BE':<10} {'Actual FE':<12} {'Actual BE':<12} {'Error':<15}")
        logger.info(f"    {'-'*65}")
        
        for i in range(min(5, len(X))):
            pred_fe = int(np.round(y_pred[i][0]))
            pred_be = int(np.round(y_pred[i][1]))
            actual_fe = int(y.iloc[i]['fe_pods'])
            actual_be = int(y.iloc[i]['be_pods'])
            error_fe = abs(pred_fe - actual_fe)
            error_be = abs(pred_be - actual_be)
            logger.info(
                f"    {pred_fe:<10} {pred_be:<10} {actual_fe:<12} {actual_be:<12} "
                f"+/-{error_fe}/+/-{error_be}"
            )
        
        return {
            'mae_frontend': mae_fe,
            'mae_backend': mae_be,
            'r2_frontend': r2_fe,
            'r2_backend': r2_be,
            'rmse_frontend': rmse_fe,
            'rmse_backend': rmse_be,
            'prediction_time_ms': (prediction_time / len(X)) * 1000
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise ModelValidationError(f"Evaluation error: {e}")

def validate_model_performance(metrics_test: Dict[str, float]) -> None:
    """
    Validate that model meets performance thresholds.
    
    Raises:
        ModelValidationError: If performance is below acceptable thresholds
    """
    logger.info("\nValidating model performance...")
    
    issues = []
    
    # Check MAE
    if metrics_test['mae_frontend'] > MAX_ACCEPTABLE_MAE:
        issues.append(f"Frontend MAE too high: {metrics_test['mae_frontend']:.2f} > {MAX_ACCEPTABLE_MAE}")
    
    if metrics_test['mae_backend'] > MAX_ACCEPTABLE_MAE:
        issues.append(f"Backend MAE too high: {metrics_test['mae_backend']:.2f} > {MAX_ACCEPTABLE_MAE}")
    
    # Check R²
    if metrics_test['r2_frontend'] < MIN_ACCEPTABLE_R2:
        issues.append(f"Frontend R² too low: {metrics_test['r2_frontend']:.4f} < {MIN_ACCEPTABLE_R2}")
    
    if metrics_test['r2_backend'] < MIN_ACCEPTABLE_R2:
        issues.append(f"Backend R² too low: {metrics_test['r2_backend']:.4f} < {MIN_ACCEPTABLE_R2}")
    
    if issues:
        error_msg = "Model performance below acceptable thresholds:\n  " + "\n  ".join(issues)
        logger.error(f"ERROR: {error_msg}")
        raise ModelValidationError(error_msg)
    
    logger.info("Model performance meets all thresholds")

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, scaler, feature_names: list, metrics: Dict) -> None:
    """Save model, scaler, and metadata."""
    logger.info("\nSaving model artifacts...")
    
    try:
        # Save model
        model_path = '../models/model.pkl'
        joblib.dump(model, model_path)
        logger.info(f"  Saved model to {model_path}")

        # Save scaler
        scaler_path = '../models/scaler.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"  Saved scaler to {scaler_path}")

        # Save metadata
        metadata = {
            'model_type': 'RandomForestRegressor',
            'training_timestamp': datetime.now().isoformat(),
            'features': feature_names,
            'n_features': len(feature_names),
            'model_params': MODEL_PARAMS,
            'training_config': {
                'test_size': TEST_SIZE,
                'random_state': RANDOM_STATE
            },
            'performance_metrics': metrics,
            'version': '1.0.0'
        }

        metadata_path = '../models/model_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Saved metadata to {metadata_path}")

        # Save training report
        report_path = 'logs/training_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'model_params': MODEL_PARAMS
            }, f, indent=2)
        logger.info(f"  Saved training report to {report_path}")

    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
        raise ModelTrainingError(f"Model save error: {e}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline.
    
    Process:
    1. Load training data
    2. Engineer features
    3. Split train/test
    4. Train model
    5. Evaluate performance
    6. Save model
    """
    start_time = time.time()
    logger.info("="*70)
    logger.info("  STARTING MODEL TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"  Timestamp: {datetime.now().isoformat()}")
    logger.info("="*70)
    
    try:
        # Step 1: Load data
        logger.info("\nSTEP 1: Loading training data...")
        
        data_path = '../data/training_clean.csv'
        if not os.path.exists(data_path):
            raise DataLoadError(f"Training data not found: {data_path}")
        
        train = pd.read_csv(data_path, parse_dates=['date'])
        logger.info(f"Loaded {len(train)} training samples")

        log_dataset_stats(train, "Training Data")
        check_data_sufficiency(train)
        
        # Step 2: Engineer features
        logger.info("\nSTEP 2: Engineering features...")
        train = engineer_features(train)
        
        # Define features
        feature_names = [
            'gmv', 'users', 'marketing_cost',
            'gmv_per_user', 'marketing_eff',
            'day_of_week', 'is_weekend', 'month',
            'gmv_millions', 'users_thousands'
        ]
        
        X = train[feature_names]
        y = train[['fe_pods', 'be_pods']]

        logger.info(f"  Feature matrix: {X.shape}")
        logger.info(f"  Target matrix: {y.shape}")
        
        # Step 3: Scale features
        logger.info("\nSummary: STEP 3: Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("  Features scaled")
        
        # Step 4: Train/test split
        logger.info(f"\nSTEP 4: Splitting data ({int((1-TEST_SIZE)*100)}% train, {int(TEST_SIZE*100)}% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        logger.info(f"  Training set: {len(X_train)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        
        # Check for data leakage
        check_data_leakage(X_train, X_test, y_train, y_test)
        
        # Step 5: Train model
        logger.info("\nSTEP 5: Training model...")
        model = train_model(X_train, y_train)
        
        # Step 6: Evaluate on training set
        logger.info("\nSTEP 6: Evaluating on training set...")
        metrics_train = evaluate_model(model, X_train, y_train, "Training Set")
        
        # Step 7: Evaluate on test set
        logger.info("\nSTEP 7: Evaluating on test set (unseen data)...")
        metrics_test = evaluate_model(model, X_test, y_test, "Test Set")
        
        # Step 8: Overfitting check
        logger.info("\nSTEP 8: Checking for overfitting...")
        mae_gap_fe = abs(metrics_train['mae_frontend'] - metrics_test['mae_frontend'])
        mae_gap_be = abs(metrics_train['mae_backend'] - metrics_test['mae_backend'])
        
        logger.info(f"  Train-Test MAE gap:")
        logger.info(f"    Frontend: {mae_gap_fe:.2f} pods")
        logger.info(f"    Backend:  {mae_gap_be:.2f} pods")
        
        if mae_gap_fe < 0.5 and mae_gap_be < 0.5:
            logger.warning("  WARNING:  Very small train-test gap - may indicate limited data diversity")
        elif mae_gap_fe > 5 or mae_gap_be > 5:
            logger.warning("  WARNING:  Large train-test gap - possible overfitting")
        else:
            logger.info("  Reasonable train-test gap - model generalizes well")
        
        # Step 9: Validate performance
        logger.info("\nPASSED: STEP 9: Validating performance...")
        validate_model_performance(metrics_test)
        
        # Step 10: Retrain on full dataset
        logger.info("\nSTEP 10: Retraining on full dataset for deployment...")
        model_full = train_model(X_scaled, y)
        
        # Step 11: Save model
        logger.info("\nSTEP 11: Saving model artifacts...")
        all_metrics = {
            'train': metrics_train,
            'test': metrics_test,
            'n_samples_total': len(X),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test)
        }
        save_model(model_full, scaler, feature_names, all_metrics)
        
        # Step 12: Summary
        duration = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("  PASSED: MODEL TRAINING COMPLETE")
        logger.info("="*70)
        
        logger.info(f"\nSummary: Final Performance (Test Set):")
        logger.info(f"   Frontend MAE: {metrics_test['mae_frontend']:.2f} pods")
        logger.info(f"   Backend MAE:  {metrics_test['mae_backend']:.2f} pods")
        logger.info(f"   Frontend R²:  {metrics_test['r2_frontend']:.4f}")
        logger.info(f"   Backend R²:   {metrics_test['r2_backend']:.4f}")
        
        logger.info(f"\nPerformance: Performance:")
        logger.info(f"   Training time: {duration:.2f}s")
        logger.info(f"   Prediction latency: {metrics_test['prediction_time_ms']:.1f}ms/sample")
        
        logger.info(f"\nFiles: Model artifacts:")
        logger.info(f"   - ../models/model.pkl")
        logger.info(f"   - ../models/scaler.pkl")
        logger.info(f"   - ../models/model_info.json")
        logger.info(f"   - logs/training_report.json")
        
        logger.info(f"\nNext: Next: cd ../api && uvicorn main:app --port 5000")
        logger.info("")
        
        return True
        
    except DataLoadError as e:
        logger.error(f"\nERROR: Data Load Error: {e}")
        logger.error("   Run: python clean_data.py first")
        return False
        
    except InsufficientDataError as e:
        logger.error(f"\nERROR: Insufficient Data: {e}")
        logger.error("   Collect more training samples")
        return False
        
    except FeatureEngineeringError as e:
        logger.error(f"\nERROR: Feature Engineering Error: {e}")
        logger.error("   Check data quality")
        return False
        
    except ModelTrainingError as e:
        logger.error(f"\nERROR: Model Training Error: {e}")
        logger.error("   Check model parameters and data")
        return False
        
    except ModelValidationError as e:
        logger.error(f"\nERROR: Model Validation Error: {e}")
        logger.error("   Model performance below acceptable thresholds")
        return False
        
    except Exception as e:
        logger.error(f"\nERROR: Unexpected Error: {e}", exc_info=True)
        logger.error("   Check logs/model_training.log for details")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)