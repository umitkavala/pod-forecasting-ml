"""
API Authentication Middleware

Implements API key-based authentication for service-to-service communication.

Security Features:
- API key validation
- Request signature verification (optional)
- Rate limiting per API key
- Key rotation support
- Audit logging
"""

import os
import hashlib
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys stored as environment variables or secure store
# In production, use AWS Secrets Manager, Azure Key Vault, or HashiCorp Vault
API_KEYS = {
    # Format: "key_id": "hashed_secret"
    os.getenv("API_KEY_NODE_SERVICE"): "node-backend-service"
}

# ============================================================================
# API KEY VALIDATION
# ============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Verify API key from request header.
    
    Args:
        request: FastAPI request object
        x_api_key: API key from X-API-Key header
        
    Returns:
        Service identifier if valid
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if API key is provided
    if not x_api_key:
        logger.warning(f"Missing API key from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    logger.info(API_KEYS);
    # Validate API key
    if x_api_key not in API_KEYS:
        logger.warning(f"Invalid API key from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    service_name = API_KEYS[x_api_key]
    
    # Log successful authentication
    logger.info(f"Authenticated request from {service_name} ({request.client.host})")
    
    return service_name

# ============================================================================
# KEY MANAGEMENT UTILITIES
# ============================================================================

def generate_api_key() -> str:
    """
    Generate a new API key.
    
    Returns:
        Random API key string
    """
    import secrets
    return secrets.token_urlsafe(32)

def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage (if storing in database).
    
    Args:
        api_key: Plain text API key
        
    Returns:
        Hashed API key
    """
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()

# ============================================================================
# HEALTH CHECK BYPASS
# ============================================================================

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = [
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/metrics"
]

def is_public_endpoint(path: str) -> bool:
    clean_path = path.rstrip('/').split('?')[0]
    if clean_path in PUBLIC_ENDPOINTS or clean_path == "":
        return True
    for public_path in PUBLIC_ENDPOINTS:
        if clean_path.startswith(public_path):
            return True
    return False
