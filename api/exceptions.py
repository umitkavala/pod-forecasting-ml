"""API-level custom exceptions.

Central place for exceptions used by the API so they can be imported in
handlers, tests, and other modules without coupling to the Observability
implementation.
"""


class ModelNotLoadedError(Exception):
    """Raised when the model has not been loaded yet."""
    pass


class PredictionError(Exception):
    """Raised when a prediction fails."""
    pass


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass
