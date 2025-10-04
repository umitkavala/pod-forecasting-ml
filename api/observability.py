from prometheus_client import Counter, Histogram, Gauge


class Observability:
    """Container for Prometheus metrics and custom exceptions.

    Metrics are created as instance attributes so the class can be instantiated
    (useful for testing or swapping implementations).
    """

    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'pod_api_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )

        self.request_duration = Histogram(
            'pod_api_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )

        self.prediction_count = Counter(
            'pod_api_predictions_total',
            'Total number of predictions made',
            ['prediction_type']
        )

        self.prediction_duration = Histogram(
            'pod_api_prediction_duration_seconds',
            'Prediction duration in seconds'
        )

        # Model metrics
        self.model_load_time = Gauge(
            'pod_api_model_load_time_seconds',
            'Time taken to load the model'
        )

        self.model_loaded = Gauge(
            'pod_api_model_loaded',
            'Whether the model is loaded (1) or not (0)'
        )

        # Error metrics
        self.error_count = Counter(
            'pod_api_errors_total',
            'Total number of errors',
            ['error_type', 'endpoint']
        )

        # Business metrics
        self.predicted_pods = Histogram(
            'pod_api_predicted_pods',
            'Distribution of predicted pod counts',
            ['pod_type']
        )

        # Authentication metrics
        self.auth_attempts = Counter(
            'pod_api_auth_attempts_total',
            'Total authentication attempts',
            ['status']
        )

        self.auth_by_service = Counter(
            'pod_api_auth_by_service_total',
            'Authentication attempts by service',
            ['service']
        )
