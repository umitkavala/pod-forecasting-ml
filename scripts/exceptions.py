"""Module containing custom exceptions for the scripts package.

"""


class ServiceAccountAuthError(Exception):
    """Raised when service account authentication fails."""
    pass


class SheetNotFoundError(Exception):
    """Raised when requested sheet is not found."""
    pass


class DataValidationError(Exception):
    """Raised when fetched data fails validation."""
    pass


class EmptyDataError(Exception):
    """Raised when fetched data is empty."""
    pass


class PermissionError(Exception):
    """Raised when service account lacks permissions."""
    pass


class DataLoadError(Exception):
    """Raised when data cannot be loaded."""
    pass

class DataQualityError(Exception):
    """Raised when data quality checks fail."""
    pass

class ColumnMissingError(Exception):
    """Raised when expected columns are missing."""
    pass

class InvalidDataTypeError(Exception):
    """Raised when data types are invalid."""
    pass

class DataLoadError(Exception):
    """Raised when training data cannot be loaded."""
    pass

class FeatureEngineeringError(Exception):
    """Raised when feature engineering fails."""
    pass

class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass

class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass

class InsufficientDataError(Exception):
    """Raised when insufficient data for training."""
    pass
