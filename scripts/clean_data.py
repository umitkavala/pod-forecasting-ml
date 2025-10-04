"""
Data cleaning with production-grade logging and error handling.

Features:
- Structured logging with timestamps
- Comprehensive exception handling
- Data quality checks
- Outlier detection
- Missing value handling
- Data type validation
"""

import os
import logging
import time
from typing import Tuple, Dict
import pandas as pd
import sys

if __name__ == "__main__":
    # Direct: python scripts/file.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.exceptions import ColumnMissingError, DataLoadError, DataQualityError, InvalidDataTypeError
else:
    # Imported as module
    from .exceptions import ColumnMissingError, DataLoadError, DataQualityError, InvalidDataTypeError


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_cleaning.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Expected columns
REQUIRED_COLUMNS = ['date', 'gmv', 'users', 'marketing_cost']
TRAINING_COLUMNS = REQUIRED_COLUMNS + ['fe_pods', 'be_pods']

# Data quality thresholds
MIN_GMV = 0
MAX_GMV = 100_000_000  # 100M
MIN_USERS = 0
MAX_USERS = 1_000_000  # 1M
MIN_MARKETING_COST = 0
MAX_MARKETING_COST = 10_000_000  # 10M
MIN_PODS = 1
MAX_PODS = 100

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def log_dataframe_info(df: pd.DataFrame, name: str) -> None:
    """Log DataFrame information."""
    logger.info(f"\n{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Data types
    logger.info(f"  Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"    {col}: {dtype}")

def check_missing_values(df: pd.DataFrame, name: str) -> Dict[str, int]:
    """
    Check for missing values.
    
    Returns:
        Dictionary of column names and missing counts
    """
    logger.info(f"\nChecking missing values in {name}...")
    
    missing = df.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()
    
    if missing_dict:
        logger.warning(f"Missing values found in {name}:")
        for col, count in missing_dict.items():
            pct = (count / len(df)) * 100
            logger.warning(f"  {col}: {count} ({pct:.1f}%)")
    else:
        logger.info(f"  No missing values in {name}")
    
    return missing_dict

def check_duplicates(df: pd.DataFrame, name: str) -> int:
    """Check for duplicate rows."""
    logger.info(f"\nChecking duplicates in {name}...")
    
    duplicates = df.duplicated().sum()
    
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows in {name}")
    else:
        logger.info(f"  No duplicates in {name}")
    
    return duplicates

def check_outliers(df: pd.DataFrame, column: str, min_val: float, max_val: float) -> Tuple[int, int]:
    """
    Check for outliers in a column.
    
    Returns:
        (count_below_min, count_above_max)
    """
    below = (df[column] < min_val).sum()
    above = (df[column] > max_val).sum()
    
    if below > 0:
        logger.warning(f"  {column}: {below} values below {min_val}")
    if above > 0:
        logger.warning(f"  {column}: {above} values above {max_val}")
    
    return below, above

def validate_data_quality(df: pd.DataFrame, name: str, has_pods: bool = False) -> None:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: DataFrame to validate
        name: Name of the dataset
        has_pods: Whether dataset includes pod columns
        
    Raises:
        DataQualityError: If critical quality issues found
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"  DATA QUALITY CHECKS: {name}")
    logger.info(f"{'='*70}")
    
    issues = []
    
    # 1. Check required columns
    logger.info("\n1. Checking required columns...")
    expected = TRAINING_COLUMNS if has_pods else REQUIRED_COLUMNS
    missing_cols = set(expected) - set(df.columns)
    
    if missing_cols:
        error_msg = f"Missing columns in {name}: {missing_cols}"
        logger.error(f"ERROR: {error_msg}")
        raise ColumnMissingError(error_msg)
    
    logger.info(f"  All required columns present")
    
    # 2. Check data types
    logger.info("\n2. Checking data types...")
    
    # Check numeric columns
    numeric_cols = ['gmv', 'users', 'marketing_cost']
    if has_pods:
        numeric_cols.extend(['fe_pods', 'be_pods'])
    
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is not numeric: {df[col].dtype}")
            # Try to convert (removing commas first)
            try:
                # Remove commas if present (e.g., "18,953,653.00" -> "18953653.00")
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"  Converted '{col}' to numeric")
            except Exception as e:
                error_msg = f"Cannot convert '{col}' to numeric: {e}"
                logger.error(error_msg)
                raise InvalidDataTypeError(error_msg)
    
    logger.info("  All numeric columns validated")
    
    # 3. Check missing values
    missing = check_missing_values(df, name)
    if missing:
        issues.append(f"Missing values: {missing}")
    
    # 4. Check duplicates
    duplicates = check_duplicates(df, name)
    if duplicates > 0:
        issues.append(f"Duplicate rows: {duplicates}")
    
    # 5. Check outliers
    logger.info(f"\n5. Checking outliers in {name}...")
    
    outliers_found = False
    
    # GMV
    below, above = check_outliers(df, 'gmv', MIN_GMV, MAX_GMV)
    if below > 0 or above > 0:
        outliers_found = True
    
    # Users
    below, above = check_outliers(df, 'users', MIN_USERS, MAX_USERS)
    if below > 0 or above > 0:
        outliers_found = True
    
    # Marketing cost
    below, above = check_outliers(df, 'marketing_cost', MIN_MARKETING_COST, MAX_MARKETING_COST)
    if below > 0 or above > 0:
        outliers_found = True
    
    # Pods
    if has_pods:
        below, above = check_outliers(df, 'fe_pods', MIN_PODS, MAX_PODS)
        if below > 0 or above > 0:
            outliers_found = True
        
        below, above = check_outliers(df, 'be_pods', MIN_PODS, MAX_PODS)
        if below > 0 or above > 0:
            outliers_found = True
    
    if not outliers_found:
        logger.info("  No outliers found")
    else:
        issues.append("Outliers detected (see warnings above)")
    
    # 6. Check for negative values
    logger.info(f"\n6. Checking for negative values in {name}...")
    
    negative_found = False
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            logger.error(f"ERROR: {col}: {negative_count} negative values")
            negative_found = True
    
    if negative_found:
        raise DataQualityError(f"{name} contains negative values in numeric columns")
    else:
        logger.info("  No negative values found")
    
    # 7. Summary
    logger.info(f"\n{'='*70}")
    if issues:
        logger.warning(f"Data quality issues in {name}:")
        for issue in issues:
            logger.warning(f"  WARNING:  {issue}")
    else:
        logger.info(f"PASSED: {name} passed all quality checks")
    logger.info(f"{'='*70}")

# ============================================================================
# MAIN CLEANING FUNCTIONS
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        pandas DataFrame
        
    Raises:
        DataLoadError: If file cannot be loaded
    """
    logger.info(f"Loading data from {filepath}...")
    
    try:
        if not os.path.exists(filepath):
            raise DataLoadError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise DataLoadError(f"File is empty: {filepath}")
        
        logger.info(f"  Loaded {len(df)} rows from {filepath}")
        return df
        
    except pd.errors.EmptyDataError:
        raise DataLoadError(f"File is empty: {filepath}")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"Failed to parse CSV: {e}")
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}", exc_info=True)
        raise DataLoadError(f"Load error: {e}")

def clean_dataframe(df: pd.DataFrame, name: str, has_pods: bool = False) -> pd.DataFrame:
    """
    Clean and validate DataFrame.
    
    Args:
        df: DataFrame to clean
        name: Name of the dataset
        has_pods: Whether dataset includes pod columns
        
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"  CLEANING {name}")
    logger.info(f"{'='*70}")
    
    df = df.copy()
    
    # 1. Parse dates
    logger.info("\n1. Parsing dates...")
    try:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        logger.info(f"  Dates parsed successfully")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    except Exception as e:
        logger.error(f"Failed to parse dates: {e}")
        raise InvalidDataTypeError(f"Date parsing failed: {e}")
    
    # 2. Convert numeric columns
    logger.info("\n2. Converting numeric columns...")
    numeric_cols = ['gmv', 'users', 'marketing_cost']
    if has_pods:
        numeric_cols.extend(['fe_pods', 'be_pods'])
    
    for col in numeric_cols:
        try:
            # Remove commas from numbers (e.g., "18,953,653.00" -> "18953653.00")
            if df[col].dtype == 'object':
                logger.info(f"  Cleaning {col} - removing commas and converting to numeric")
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Check for NaN values after conversion
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                logger.warning(f"  {col}: {nan_count} values became NaN after conversion")
            else:
                logger.info(f"  {col} converted to numeric successfully")
                
        except Exception as e:
            logger.error(f"Failed to convert {col}: {e}")
            raise InvalidDataTypeError(f"Numeric conversion failed for {col}: {e}")
    
    # 3. Handle missing values
    logger.info("\n3. Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    if missing_before > 0:
        logger.warning(f"Found {missing_before} missing values")
        
        # Show which columns have missing values
        missing_cols = df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0]
        for col, count in missing_cols.items():
            logger.warning(f"  {col}: {count} missing values")
        
        # Drop rows with missing values
        rows_before = len(df)
        df = df.dropna()
        rows_dropped = rows_before - len(df)
        
        logger.info(f"  Dropped {rows_dropped} rows with missing values")
        logger.info(f"  {len(df)} rows remaining")
    else:
        logger.info("  No missing values to handle")
    
    # 4. Remove duplicates
    logger.info("\n4. Removing duplicates...")
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = duplicates_before - len(df)
    
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate rows")
    else:
        logger.info("  No duplicates to remove")
    
    # 5. Sort by date
    logger.info("\n5. Sorting by date...")
    df = df.sort_values('date').reset_index(drop=True)
    logger.info("  Data sorted by date")
    
    # 6. Validate data quality
    validate_data_quality(df, name, has_pods)
    
    return df

def main():
    """
    Main function to clean data.
    
    Process:
    1. Load raw data files
    2. Clean and validate Historical data
    3. Clean and validate Budget data
    4. Save cleaned data
    """
    start_time = time.time()
    logger.info("="*70)
    logger.info("  STARTING DATA CLEANING")
    logger.info("="*70)
    
    try:
        # Step 1: Load Historical Data
        logger.info("\nSTEP STEP 1: Loading Historical Data")
        historical_df = load_data('../data/historical_raw.csv')
        log_dataframe_info(historical_df, "Historical (Raw)")
        
        # Step 2: Clean Historical Data
        logger.info("\nSTEP STEP 2: Cleaning Historical Data")
        historical_clean = clean_dataframe(historical_df, "Historical", has_pods=True)
        log_dataframe_info(historical_clean, "Historical (Clean)")
        
        # Step 3: Load Budget Data
        logger.info("\nSTEP STEP 3: Loading Budget Data")
        budget_df = load_data('../data/budget_raw.csv')
        log_dataframe_info(budget_df, "Budget (Raw)")
        
        # Step 4: Clean Budget Data
        logger.info("\nSTEP STEP 4: Cleaning Budget Data")
        budget_clean = clean_dataframe(budget_df, "Budget", has_pods=False)
        log_dataframe_info(budget_clean, "Budget (Clean)")
        
        # Step 5: Save cleaned data
        logger.info("\nSTEP STEP 5: Saving Cleaned Data")
        
        # Save training data (Historical)
        training_path = '../data/training_clean.csv'
        historical_clean.to_csv(training_path, index=False)
        logger.info(f"  Saved training data to {training_path}")
        
        # Save budget data
        budget_path = '../data/budget_clean.csv'
        budget_clean.to_csv(budget_path, index=False)
        logger.info(f"  Saved budget data to {budget_path}")
        
        # Step 6: Summary
        duration = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("  PASSED: DATA CLEANING COMPLETE")
        logger.info("="*70)
        
        logger.info(f"\nSummary: Summary:")
        logger.info(f"   Training data:  {len(historical_clean)} rows, {len(historical_clean.columns)} columns")
        logger.info(f"   Budget data:    {len(budget_clean)} rows, {len(budget_clean.columns)} columns")
        logger.info(f"   Duration:       {duration:.2f} seconds")
        
        logger.info(f"\nFiles: Files saved:")
        logger.info(f"   - {training_path}")
        logger.info(f"   - {budget_path}")
        
        logger.info(f"\nQuality: Data Quality:")
        logger.info(f"     All dates parsed successfully")
        logger.info(f"     All numeric conversions successful")
        logger.info(f"     No missing values")
        logger.info(f"     No duplicates")
        logger.info(f"     Data sorted by date")
        logger.info(f"     All quality checks passed")
        
        logger.info(f"\nNext: Next: python train_model.py")
        logger.info("")
        
        return True
        
    except DataLoadError as e:
        logger.error(f"\nERROR: Data Load Error: {e}")
        logger.error("   Check that raw data files exist")
        return False
        
    except ColumnMissingError as e:
        logger.error(f"\nERROR: Column Missing: {e}")
        logger.error("   Check data structure in Google Sheets")
        return False
        
    except InvalidDataTypeError as e:
        logger.error(f"\nERROR: Invalid Data Type: {e}")
        logger.error("   Check data types in source")
        return False
        
    except DataQualityError as e:
        logger.error(f"\nERROR: Data Quality Error: {e}")
        logger.error("   Review quality check warnings above")
        return False
        
    except Exception as e:
        logger.error(f"\nERROR: Unexpected Error: {e}", exc_info=True)
        logger.error("   Check logs/data_cleaning.log for details")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)