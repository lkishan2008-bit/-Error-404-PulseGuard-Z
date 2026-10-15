"""
==============================================================================
PulseGuard-Z  -  Backend Authentication & Security
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : backend/auth.py

Purpose
-------
Provides dual authentication for the PulseGuard-Z FastAPI backend:
1. In-memory session authentication for dashboard operators (username/password login)
   with cryptographically secure PBKDF2-HMAC-SHA256 password hashing, constant-time
   verification, in-memory token management, and IP-based rate limiting.
2. API Key authentication (X-API-Key header, Authorization Bearer, or query parameter)
   for automated agents and backend pipelines.

Security Guarantees:
- Passwords are never stored in plaintext; salted PBKDF2-HMAC-SHA256 (260,000 iterations).
- Constant-time comparison via secrets.compare_digest prevents timing attacks.
- Sensitive credentials, hashes, and tokens are NEVER logged.
- In-memory session store is thread-safe and enforces automatic TTL expiration.
- Failed login rate-limiting: 5 failed attempts per client IP in a 5-minute rolling window.
==============================================================================
"""

import hashlib
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

# ---------------------------------------------------------------------------
# Configuration & Environment Loading
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
if ENV_PATH.is_file():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Legacy API Key settings
CONFIGURED_API_KEY = os.getenv("API_KEY", "").strip()
REQUIRE_API_KEY_ENV = os.getenv("REQUIRE_API_KEY", "").strip().lower()

# Dashboard Session Auth settings
DASHBOARD_USER = os.getenv("DASHBOARD_USER", "").strip()
DASHBOARD_PASSWORD_HASH = os.getenv("DASHBOARD_PASSWORD_HASH", "").strip()

try:
    SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "30"))
except ValueError:
    SESSION_TTL_MINUTES = 30

PLACEHOLDERS = {
    "change-me-in-your-local-env",
    "your-api-key-here",
    "placeholder_salt_hex:placeholder_hash_hex",
    "",
}

PBKDF2_ITERATIONS = 260_000


# ---------------------------------------------------------------------------
# Password Hashing & Verification (PBKDF2-HMAC-SHA256)
# ---------------------------------------------------------------------------
def hash_password(password: str, salt_hex: Optional[str] = None) -> str:
    """
    Computes a PBKDF2-HMAC-SHA256 hash using 260,000 iterations and a 16-byte salt.
    Format returned: salt_hex:hash_hex
    """
    if salt_hex is None:
        salt_bytes = secrets.token_bytes(16)
        salt_hex = salt_bytes.hex()
    else:
        salt_bytes = bytes.fromhex(salt_hex)

    hash_bytes = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS,
    )
    return f"{salt_hex}:{hash_bytes.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Constant-time verification of password against stored salt_hex:hash_hex.
    Prevents timing side-channel attacks.
    """
    if not stored_hash or ":" not in stored_hash:
        return False
    try:
        salt_hex, expected_hash_hex = stored_hash.split(":", 1)
        salt_bytes = bytes.fromhex(salt_hex)
        computed_hash_bytes = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_bytes,
            PBKDF2_ITERATIONS,
        )
        return secrets.compare_digest(computed_hash_bytes.hex(), expected_hash_hex)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# In-Memory Session Store
# ---------------------------------------------------------------------------
class SessionStore:
    """
    Thread-safe in-memory session store with TTL expiration.
    Suitable for single-instance hackathon deployments.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_session(
        self, username: str, ttl_minutes: int = SESSION_TTL_MINUTES
    ) -> Tuple[str, datetime]:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=ttl_minutes)

        with self._lock:
            # Purge expired sessions
            expired = [k for k, v in self._sessions.items() if v["expires_at"] <= now]
            for k in expired:
                self._sessions.pop(k, None)

            self._sessions[token] = {
                "username": username,
                "expires_at": expires_at,
                "created_at": now,
            }
        return token, expires_at

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._lock:
            session = self._sessions.get(token)
            if not session:
                return None
            if datetime.now(timezone.utc) > session["expires_at"]:
                self._sessions.pop(token, None)
                return None
            return session

    def revoke_session(self, token: str) -> bool:
        if not token:
            return False
        with self._lock:
            return self._sessions.pop(token, None) is not None

    def active_sessions_count(self) -> int:
        now = datetime.now(timezone.utc)
        with self._lock:
            return sum(1 for v in self._sessions.values() if v["expires_at"] > now)


session_store = SessionStore()


# ---------------------------------------------------------------------------
# Rate Limiting (5 failed attempts per IP in 5 minutes)
# ---------------------------------------------------------------------------
class LoginRateLimiter:
    """
    Thread-safe rate limiter tracking failed login attempts per client IP.
    """

    def __init__(self, max_attempts: int = 5, window_seconds: int = 300):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def is_rate_limited(self, ip: str) -> bool:
        with self._lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=self.window_seconds)
            history = [t for t in self._attempts.get(ip, []) if t > window_start]
            self._attempts[ip] = history
            return len(history) >= self.max_attempts

    def record_failure(self, ip: str):
        with self._lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=self.window_seconds)
            history = [t for t in self._attempts.get(ip, []) if t > window_start]
            history.append(now)
            self._attempts[ip] = history

    def record_success(self, ip: str):
        with self._lock:
            self._attempts.pop(ip, None)


login_rate_limiter = LoginRateLimiter(max_attempts=5, window_seconds=300)


# ---------------------------------------------------------------------------
# Authentication Status & Introspection
# ---------------------------------------------------------------------------
def is_auth_enabled() -> bool:
    """
    Returns True if authentication is actively enforced.
    Auth is enforced if REQUIRE_API_KEY is true, or if valid API_KEY is set,
    or if DASHBOARD_USER and DASHBOARD_PASSWORD_HASH are configured.
    """
    if REQUIRE_API_KEY_ENV in {"1", "true", "yes"}:
        return True
    has_api_key = bool(CONFIGURED_API_KEY and CONFIGURED_API_KEY not in PLACEHOLDERS)
    has_dashboard_auth = bool(
        DASHBOARD_USER and DASHBOARD_PASSWORD_HASH and DASHBOARD_PASSWORD_HASH not in PLACEHOLDERS
    )
    return has_api_key or has_dashboard_auth


# ---------------------------------------------------------------------------
# FastAPI Dependency & Security Schemes
# ---------------------------------------------------------------------------
api_key_header_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API Key provided via X-API-Key header",
)
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Bearer token (Session token or API Key) provided via Authorization header",
)


TRUSTED_PROXIES = {
    ip.strip() for ip in os.getenv("TRUSTED_PROXIES", "").split(",") if ip.strip()
}


def get_client_ip(request: Request) -> str:
    """
    Safely resolves the client IP address for rate limiting.
    Does NOT blindly trust X-Forwarded-For headers to prevent IP spoofing attacks.
    Only inspects X-Forwarded-For if request.client.host is explicitly declared
    in the TRUSTED_PROXIES allowlist. Otherwise falls back to direct peer IP.
    """
    peer_ip = request.client.host if request.client and request.client.host else "127.0.0.1"
    if TRUSTED_PROXIES and peer_ip in TRUSTED_PROXIES:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_candidate = forwarded.split(",")[0].strip()
            if client_candidate:
                return client_candidate
    return peer_ip


def get_current_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_creds: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[str]:
    """
    Extracts authentication credentials from:
    1. X-API-Key header
    2. Bearer token in Authorization header
    (Note: Query parameter ?api_key= is intentionally rejected to prevent URL/log credential exposure.)
    """
    if header_key:
        return header_key.strip()
    if bearer_creds and bearer_creds.credentials:
        return bearer_creds.credentials.strip()
    return None


def verify_api_key(
    request: Request,
    credential: Optional[str] = Security(get_current_api_key),
) -> Optional[str]:
    """
    FastAPI dependency to verify authentication for protected endpoints.
    Accepts:
    1. A valid, unexpired Bearer session token from /auth/login.
    2. A valid CONFIGURED_API_KEY (via X-API-Key or Authorization Bearer header).
    3. If auth is not configured (open development mode), allows access anonymously.
    """
    auth_active = is_auth_enabled()

    # 1. Check if credential matches an active in-memory session token
    if credential:
        session = session_store.get_session(credential)
        if session:
            return f"session:{session['username']}"

    # 2. Check if credential matches the configured static API key
    if credential and CONFIGURED_API_KEY and CONFIGURED_API_KEY not in PLACEHOLDERS:
        if secrets.compare_digest(credential, CONFIGURED_API_KEY):
            return "api-key-client"

    # 3. Development open mode (neither API key nor dashboard auth enforced)
    if not auth_active:
        return credential or "dev-mode-anonymous"

    # 4. Auth is active, but credential was missing or invalid
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a valid Bearer session token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired session token / API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_auth_status() -> dict:
    """
    Returns current auth configuration status for introspection.
    Never exposes passwords, hashes, or secret keys.
    """
    enabled = is_auth_enabled()
    has_api_key = bool(CONFIGURED_API_KEY and CONFIGURED_API_KEY not in PLACEHOLDERS)
    has_dashboard = bool(
        DASHBOARD_USER and DASHBOARD_PASSWORD_HASH and DASHBOARD_PASSWORD_HASH not in PLACEHOLDERS
    )
    return {
        "auth_enabled": enabled,
        "mode": "enforced" if enabled else "development_open",
        "api_key_configured": has_api_key,
        "dashboard_user_configured": has_dashboard,
        "session_ttl_minutes": SESSION_TTL_MINUTES,
        "active_sessions": session_store.active_sessions_count(),
        "rate_limiting": "5 failures per 5 minutes per IP",
        "trusted_proxies_configured": bool(TRUSTED_PROXIES),
        "header_supported": "X-API-Key",
        "bearer_supported": "Authorization: Bearer <session_token_or_api_key>",
        "query_param_supported": "disabled (security hardening: header-only authentication)",
    }
