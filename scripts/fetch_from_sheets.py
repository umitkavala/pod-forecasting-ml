"""
Fetch data from Google Sheets using Service Account (for cronjobs).

Features:
- Service Account authentication (no user interaction)
- Suitable for automated cronjobs
- Structured logging with timestamps
- Comprehensive exception handling
- Validation of fetched data
- Retry logic for API calls
- Data integrity checks
"""

import os
import logging
import json
import sys
import time
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
load_dotenv()

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
        logging.FileHandler('logs/data_ingestion.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# CUSTOM EXCEPTIONS (from scripts.exceptions)
# ============================================================================
# Works BOTH as direct script AND as module!
if __name__ == "__main__":
    # Direct: python scripts/file.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.exceptions import (
        ServiceAccountAuthError,
        SheetNotFoundError,
        DataValidationError,
        EmptyDataError,
        PermissionError,
    )
else:
    from .exceptions import (
            ServiceAccountAuthError,
            SheetNotFoundError,
            DataValidationError,
            EmptyDataError,
            PermissionError,
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
print(SERVICE_ACCOUNT_FILE)

# Expected columns for validation
EXPECTED_HISTORICAL_COLUMNS = ['date', 'gmv', 'users', 'marketing_cost', 'fe_pods', 'be_pods']
EXPECTED_BUDGET_COLUMNS = ['date', 'gmv', 'users', 'marketing_cost']

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_dataframe(df: pd.DataFrame, expected_columns: List[str], sheet_name: str) -> None:
    """
    Validate that dataframe has expected structure.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        sheet_name: Name of the sheet (for error messages)
        
    Raises:
        DataValidationError: If validation fails
        EmptyDataError: If dataframe is empty
    """
    logger.info(f"Validating {sheet_name} data...")
    
    # Check if empty
    if df.empty:
        raise EmptyDataError(f"{sheet_name} data is empty")
    
    # Check columns
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"{sheet_name} missing columns: {missing_columns}. "
            f"Expected: {expected_columns}, Got: {list(df.columns)}"
        )
    
    # Check for null values in critical columns
    null_counts = df[expected_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"{sheet_name} has null values:\n{null_counts[null_counts > 0]}")
    
    # Check data types
    numeric_columns = [col for col in expected_columns if col != 'date']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"{sheet_name}: Column '{col}' is not numeric, will attempt conversion")
    
    logger.info(f"  {sheet_name} validation passed: {len(df)} rows, {len(df.columns)} columns")

def retry_on_error(func):
    """Decorator to retry function on error."""
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                if attempt == MAX_RETRIES:
                    logger.error(f"Failed after {MAX_RETRIES} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt} failed, retrying in {RETRY_DELAY}s: {e}")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                raise
    return wrapper

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def authenticate_service_account() -> object:
    """
    Authenticate using Service Account credentials.
    
    Returns:
        Google Sheets service object
        
    Raises:
        ServiceAccountAuthError: If authentication fails
    """
    logger.info("Starting Service Account authentication...")
    
    try:
        # Check if service account file exists
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise ServiceAccountAuthError(
                f"Service account file not found: {SERVICE_ACCOUNT_FILE}\n"
                f"Please download from Google Cloud Console:\n"
                f"1. Go to IAM & Admin > Service Accounts\n"
                f"2. Create or select service account\n"
                f"3. Create key (JSON)\n"
                f"4. Save as {SERVICE_ACCOUNT_FILE}"
            )
        
        logger.info(f"Loading credentials from {SERVICE_ACCOUNT_FILE}")
        
        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )
        
        # Build service
        service = build('sheets', 'v4', credentials=credentials)
        
        logger.info("  Service Account authentication successful")
        logger.info(f"  Service account: {credentials.service_account_email}")
        
        return service
        
    except FileNotFoundError:
        raise ServiceAccountAuthError(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
    except json.JSONDecodeError as e:
        raise ServiceAccountAuthError(f"Invalid JSON in service account file: {e}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}", exc_info=True)
        raise ServiceAccountAuthError(f"Authentication error: {e}")

@retry_on_error
def fetch_sheet_data(service, sheet_name: str) -> List[List]:
    """
    Fetch data from a specific sheet with retry logic.
    
    Args:
        service: Google Sheets service object
        sheet_name: Name of the sheet to fetch
        
    Returns:
        List of rows from the sheet
        
    Raises:
        SheetNotFoundError: If sheet doesn't exist
        PermissionError: If service account lacks access
        HttpError: If API call fails after retries
    """
    logger.info(f"Fetching data from sheet: {sheet_name}")
    
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=sheet_name
        ).execute()
        
        values = result.get('values', [])
        
        if not values:
            logger.warning(f"Sheet '{sheet_name}' is empty")
            raise EmptyDataError(f"No data found in sheet '{sheet_name}'")
        
        logger.info(f"  Fetched {len(values)} rows from {sheet_name}")
        return values
        
    except HttpError as e:
        if e.resp.status == 404:
            raise SheetNotFoundError(f"Sheet '{sheet_name}' not found in spreadsheet")
        elif e.resp.status == 403:
            raise PermissionError(
                f"Service account lacks permission to access spreadsheet.\n"
                f"Please share the spreadsheet with the service account email:\n"
                f"  IAM & Admin > Service Accounts > Copy email\n"
                f"  Then share the Google Sheet with that email (Viewer access)"
            )
        raise

def convert_to_dataframe(values: List[List], sheet_name: str) -> pd.DataFrame:
    """
    Convert sheet values to pandas DataFrame.
    
    Args:
        values: List of rows from Google Sheets
        sheet_name: Name of the sheet (for logging)
        
    Returns:
        pandas DataFrame
        
    Raises:
        DataValidationError: If conversion fails
    """
    logger.info(f"Converting {sheet_name} to DataFrame...")
    
    try:
        # First row is headers
        headers = values[0]
        data = values[1:]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Convert column names to lowercase and replace spaces
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        logger.info(f"  Created DataFrame with columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to convert {sheet_name} to DataFrame: {e}", exc_info=True)
        raise DataValidationError(f"DataFrame conversion failed: {e}")

def main():
    """
    Main function to fetch data from Google Sheets.
    
    Process:
    1. Authenticate with Service Account
    2. Fetch Historical sheet
    3. Fetch Budget sheet
    4. Validate data
    5. Save to CSV files
    
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    logger.info("="*70)
    logger.info("  STARTING DATA INGESTION FROM GOOGLE SHEETS (SERVICE ACCOUNT)")
    logger.info("="*70)
    logger.info(f"  Timestamp: {datetime.now().isoformat()}")
    logger.info(f"  Spreadsheet ID: {SPREADSHEET_ID}")
    logger.info("="*70)
    
    try:
        # Create data directory
        os.makedirs('../data', exist_ok=True)
        logger.info("  Data directory ready")
        
        # Step 1: Authenticate with Service Account
        logger.info("\n1. Authenticating with Service Account...")
        service = authenticate_service_account()
        
        # Step 2: Fetch Historical Data
        logger.info("\n2. Fetching Historical data...")
        try:
            historical_values = fetch_sheet_data(service, 'Historical')
            historical_df = convert_to_dataframe(historical_values, 'Historical')
            validate_dataframe(historical_df, EXPECTED_HISTORICAL_COLUMNS, 'Historical')
            
            # Save to CSV
            historical_path = '../data/historical_raw.csv'
            historical_df.to_csv(historical_path, index=False)
            logger.info(f"  Saved Historical data to {historical_path}")
            
        except (SheetNotFoundError, EmptyDataError, DataValidationError) as e:
            logger.error(f"Failed to fetch Historical data: {e}")
            raise
        
        # Step 3: Fetch Budget Data
        logger.info("\n3. Fetching Budget data...")
        try:
            budget_values = fetch_sheet_data(service, 'Budget')
            budget_df = convert_to_dataframe(budget_values, 'Budget')
            validate_dataframe(budget_df, EXPECTED_BUDGET_COLUMNS, 'Budget')
            
            # Save to CSV
            budget_path = '../data/budget_raw.csv'
            budget_df.to_csv(budget_path, index=False)
            logger.info(f"  Saved Budget data to {budget_path}")
            
        except (SheetNotFoundError, EmptyDataError, DataValidationError) as e:
            logger.error(f"Failed to fetch Budget data: {e}")
            raise
        
        # Step 4: Summary
        duration = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("  DATA INGESTION COMPLETE")
        logger.info("="*70)
        logger.info(f"\nSummary:")
        logger.info(f"   Historical: {len(historical_df)} rows, {len(historical_df.columns)} columns")
        logger.info(f"   Budget:     {len(budget_df)} rows, {len(budget_df.columns)} columns")
        logger.info(f"   Duration:   {duration:.2f} seconds")
        logger.info(f"\nFiles saved:")
        logger.info(f"   - {historical_path}")
        logger.info(f"   - {budget_path}")
        logger.info(f"\nNext: python clean_data.py")
        logger.info("")
        
        return True
        
    except ServiceAccountAuthError as e:
        logger.error(f"\nERROR: Service Account Authentication Failed: {e}")
        logger.error("   Check service-account-key.json exists and is valid")
        return False
        
    except PermissionError as e:
        logger.error(f"\nERROR: Permission Denied: {e}")
        logger.error("   Share the Google Sheet with your service account email")
        return False
        
    except SheetNotFoundError as e:
        logger.error(f"\nERROR: Sheet Not Found: {e}")
        logger.error("   Verify sheet names in Google Sheets")
        return False
        
    except EmptyDataError as e:
        logger.error(f"\nERROR: Empty Data: {e}")
        logger.error("   Check that sheets contain data")
        return False
        
    except DataValidationError as e:
        logger.error(f"\nERROR: Data Validation Error: {e}")
        logger.error("   Check column names and data structure")
        return False
        
    except Exception as e:
        logger.error(f"\nERROR: Unexpected Error: {e}", exc_info=True)
        logger.error("   Check logs/data_ingestion.log for details")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)