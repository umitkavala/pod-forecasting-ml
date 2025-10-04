"""
Pod Forecasting FastAPI Application - WITH WORKING FILE LOGGING

Key fix: Logging configured BEFORE app creation, not inside lifespan
"""

import os
import time
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest


from .models import (
    BatchPredictionRequest, BatchPredictionResponse, 
    HealthResponse, PredictionRequest, PredictionResponse
)
from .predictor import PodPredictor
from .auth import verify_api_key, is_public_endpoint, API_KEYS
from .observability import Observability
from .exceptions import ModelNotLoadedError, PredictionError, ValidationError

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION - BEFORE EVERYTHING ELSE!
# ============================================================================

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create logs directory
LOGS_DIR = Path(__file__).parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / 'pod_api.log'

# Create handlers with explicit flush
class FlushFileHandler(logging.FileHandler):
    """File handler that flushes after every write."""
    def emit(self, record):
        super().emit(record)
        self.flush()

file_handler = FlushFileHandler(str(LOG_FILE), mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Get app logger
logger = logging.getLogger(__name__)

# ============================================================================
# OBSERVABILITY
# ============================================================================

observability = Observability()

request_count = observability.request_count
request_duration = observability.request_duration
prediction_count = observability.prediction_count
prediction_duration = observability.prediction_duration
model_load_time = observability.model_load_time
model_loaded = observability.model_loaded
error_count = observability.error_count
predicted_pods = observability.predicted_pods
auth_attempts = observability.auth_attempts
auth_by_service = observability.auth_by_service

# ============================================================================
# GLOBAL PREDICTOR
# ============================================================================

predictor: Optional[PodPredictor] = None

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler."""
    global predictor

    # Startup
    logger.info("="*70)
    logger.info("Starting Pod Forecasting API with Authentication")
    logger.info("="*70)
    
    if not API_KEYS:
        logger.warning("WARNING: No API keys configured!")
    else:
        logger.info(f"Authentication configured with {len(API_KEYS)} API keys")

    try:
        start_time = time.time()
        predictor = PodPredictor()
        load_time = time.time() - start_time

        model_load_time.set(load_time)
        model_loaded.set(1)

        logger.info(f"Model loaded in {load_time:.2f}s")
        logger.info("API ready")
        logger.info("="*70)

    except Exception as e:
        logger.critical(f"Failed to load model: {e}", exc_info=True)
        model_loaded.set(0)
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")
    predictor = None
    model_loaded.set(0)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Pod Forecasting API",
    description="Production-ready ML API with authentication",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def process_request(request: Request, call_next):
    """Combined middleware with file logging."""
    request_id = f"{int(time.time() * 1000)}-{id(request)}"
    start_time = time.time()
    
    # Log to console AND file
    log_msg = f"-> [{request_id}] {request.method} {request.url.path}"
    print(log_msg)  # Console
    logger.info(log_msg)  # File + Console
    
    # Check if public endpoint
    if is_public_endpoint(request.url.path):
        log_msg = f"  Public endpoint - no auth required"
        print(log_msg)
        logger.info(log_msg)
        response = await call_next(request)
    else:
        # Protected endpoint
        x_api_key = request.headers.get("X-API-Key")
        try:
            service = await verify_api_key(request, x_api_key)
            log_msg = f"  Authenticated: {service}"
            print(log_msg)
            logger.info(log_msg)

            auth_attempts.labels(status='success').inc()
            auth_by_service.labels(service=service).inc()

            request.state.service = service
            request.state.request_id = request_id

            response = await call_next(request)

        except HTTPException as e:
            log_msg = f"  Auth failed: {e.detail}"
            print(log_msg)
            logger.warning(log_msg)
            auth_attempts.labels(status='failed').inc()

            response = JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "request_id": request_id},
                headers=e.headers or {}
            )
    
    # Track metrics
    duration = time.time() - start_time
    status_code = response.status_code
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=status_code
    ).inc()
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.3f}"
    
    # Log completion
    log_msg = f"<- [{request_id}] {status_code} in {duration*1000:.2f}ms"
    print(log_msg)
    logger.info(log_msg)
    
    return response

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    request_id = getattr(request.state, 'request_id', 'UNKNOWN')
    logger.error(f"Model not loaded: {exc}")
    error_count.labels(error_type='ModelNotLoadedError', endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Model not loaded", "request_id": request_id}
    )

@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    request_id = getattr(request.state, 'request_id', 'UNKNOWN')
    logger.error(f"Prediction error: {exc}", exc_info=True)
    error_count.labels(error_type='PredictionError', endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Prediction failed", "message": str(exc), "request_id": request_id}
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    request_id = getattr(request.state, 'request_id', 'UNKNOWN')
    logger.warning(f"Validation error: {exc}")
    error_count.labels(error_type='ValidationError', endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation failed", "message": str(exc), "request_id": request_id}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, 'request_id', 'UNKNOWN')
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    error_count.labels(error_type=type(exc).__name__, endpoint=request.url.path).inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "request_id": request_id}
    )

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Pod Forecasting API",
        "version": "1.0.0",
        "authentication": "API key required (X-API-Key header)",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "predict": "POST /predict",
            "batch": "POST /forecast/batch"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if predictor is None:
        raise ModelNotLoadedError("Model is not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_info": predictor.get_model_info()
    }

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest, req: Request):
    request_id = getattr(req.state, 'request_id', 'UNKNOWN')
    service = getattr(req.state, 'service', 'UNKNOWN')
    
    if predictor is None:
        raise ModelNotLoadedError("Model is not loaded")
    
    try:
        logger.info(f"[{request_id}] Prediction from {service}: date={request.date}")
        
        if request.gmv <= 0 or request.users <= 0 or request.marketing_cost <= 0:
            raise ValidationError("All numeric inputs must be positive")
        
        start_time = time.time()
        date_obj = datetime.combine(request.date, datetime.min.time())
        
        result = predictor.predict(
            date=date_obj,
            gmv=request.gmv,
            users=request.users,
            marketing_cost=request.marketing_cost
        )
        
        duration = time.time() - start_time
        prediction_duration.observe(duration)
        prediction_count.labels(prediction_type='single').inc()
        
        predicted_pods.labels(pod_type='frontend').observe(result["predictions"]["frontend_pods"])
        predicted_pods.labels(pod_type='backend').observe(result["predictions"]["backend_pods"])
        
        logger.info(
            f"[{request_id}] FE={result['predictions']['frontend_pods']}, "
            f"BE={result['predictions']['backend_pods']} ({duration*1000:.2f}ms)"
        )
        
        return {
            "date": request.date,
            "input": {
                "gmv": request.gmv,
                "users": request.users,
                "marketing_cost": request.marketing_cost
            },
            **result
        }
        
    except (ModelNotLoadedError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Prediction failed: {e}", exc_info=True)
        raise PredictionError(f"Prediction failed: {str(e)}")

@app.post("/forecast/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest, req: Request):
    request_id = getattr(req.state, 'request_id', 'UNKNOWN')
    service = getattr(req.state, 'service', 'UNKNOWN')
    
    if predictor is None:
        raise ModelNotLoadedError("Model is not loaded")
    
    try:
        batch_size = len(request.predictions)
        logger.info(f"[{request_id}] Batch from {service}: {batch_size} items")
        
        start_time = time.time()
        results = []
        
        for pred_request in request.predictions:
            try:
                date_obj = datetime.combine(pred_request.date, datetime.min.time())
                result = predictor.predict(
                    date=date_obj,
                    gmv=pred_request.gmv,
                    users=pred_request.users,
                    marketing_cost=pred_request.marketing_cost
                )
                
                results.append({
                    "date": pred_request.date.isoformat(),
                    "predictions": result["predictions"],
                    "confidence_intervals": result["confidence_intervals"]
                })
                
                predicted_pods.labels(pod_type='frontend').observe(result["predictions"]["frontend_pods"])
                predicted_pods.labels(pod_type='backend').observe(result["predictions"]["backend_pods"])
            except Exception as e:
                logger.error(f"[{request_id}] Batch item failed: {e}")
                continue
        
        duration = time.time() - start_time
        prediction_count.labels(prediction_type='batch').inc()

        logger.info(f"[{request_id}] Batch: {len(results)}/{batch_size} ({duration:.2f}s)")

        return {"count": len(results), "predictions": results}

    except (ModelNotLoadedError, ValidationError):
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Batch failed: {e}", exc_info=True)
        raise PredictionError(f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    if predictor is None:
        raise ModelNotLoadedError("Model is not loaded")
    return predictor.get_model_info()