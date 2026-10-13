"""
==============================================================================
PulseGuard-Z  -  Backend Authentication & Security
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : backend/auth.py

Purpose
-------
Provides API key authentication for the PulseGuard-Z FastAPI backend.
Reads configuration from environment variables or .env located in the project root.
Supports X-API-Key header, Authorization Bearer token, and optional query parameter.
In development mode (no API key set or default placeholder), requests proceed with
a transparent warning to ease local development and testing.
==============================================================================
"""

import os
import secrets
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

# Locate project root and load .env if present
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Configuration
CONFIGURED_API_KEY = os.getenv("API_KEY", "").strip()
REQUIRE_API_KEY_ENV = os.getenv("REQUIRE_API_KEY", "").strip().lower()

# Placeholder markers from .env.example
PLACEHOLDERS = {"change-me-in-your-local-env", "your-api-key-here", ""}

# Decide if authentication is actively enforced
def is_auth_enabled() -> bool:
    """
    Returns True if an active, non-placeholder API key is set or REQUIRE_API_KEY=true.
    """
    if REQUIRE_API_KEY_ENV in {"1", "true", "yes"}:
        return True
    return bool(CONFIGURED_API_KEY and CONFIGURED_API_KEY not in PLACEHOLDERS)


# Security schemes for OpenAPI documentation
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, description="API Key provided via X-API-Key header")
bearer_scheme = HTTPBearer(auto_error=False, description="API Key provided via Bearer token")


def get_current_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[str]:
    """
    Extracts the API key from:
    1. X-API-Key header
    2. Bearer token in Authorization header
    3. api_key query parameter
    """
    if header_key:
        return header_key
    if bearer_creds and bearer_creds.credentials:
        return bearer_creds.credentials
    # Fallback to query param
    query_key = request.query_params.get("api_key")
    if query_key:
        return query_key
    return None


def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(get_current_api_key),
) -> Optional[str]:
    """
    FastAPI dependency to verify the API key.
    - If authentication is disabled (dev mode), allows all requests.
    - If authentication is enabled, enforces valid key comparison using secrets.compare_digest.
    """
    auth_active = is_auth_enabled()

    if not auth_active:
        # Development mode: auth is optional
        return api_key or "dev-mode-anonymous"

    expected_key = CONFIGURED_API_KEY
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configured to require API key, but no valid API_KEY is set in environment.",
        )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via 'X-API-Key' header, 'Authorization: Bearer <key>', or '?api_key=<key>'.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def get_auth_status() -> dict:
    """
    Returns current auth configuration status for introspection.
    """
    enabled = is_auth_enabled()
    return {
        "auth_enabled": enabled,
        "mode": "enforced" if enabled else "development_open",
        "header_supported": "X-API-Key",
        "bearer_supported": "Authorization: Bearer <token>",
        "query_param_supported": "api_key",
    }
