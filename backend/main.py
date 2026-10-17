"""
==============================================================================
PulseGuard-Z  -  Phase 3: Operational FastAPI Backend
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : backend/main.py

Purpose
-------
Provides a reliable, high-performance REST API for the PulseGuard-Z flash fraud
detection platform on IBM LinuxONE. Serves dashboard analytics, paginated and
filterable transactions, suspicious account queues ranked by TGNN risk score,
deep account profiles, and signal explanations.

Data Source & Fidelity Notice
-----------------------------
This backend loads data generated in Phase 1 and scored in Phase 2:
- data/accounts.csv
- data/transactions.csv
- data/scores.csv
All metrics and responses are computed strictly from these local CSV files.
The data represents a synthetic benchmark simulating Government-to-Person (G2P)
welfare flash fraud on IBM LinuxONE mainframes.

Risk Score vs Probability
-------------------------
The `tgnn_score` output by the Temporal Graph Neural Network represents a
topological risk score, not a calibrated real-world probability of fraud.
Due to severe class imbalance in social welfare payments (~0.35% fraud prior),
raw scores peak around ~0.29. The default calibrated decision threshold is 0.20
(configurable via the `threshold` query parameter or DEFAULT_RISK_THRESHOLD env var).
==============================================================================
"""

import math
import os
import sys
import typing
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python 3.14 pre-release compatibility shim for private typing._eval_type
if sys.version_info >= (3, 14) and not getattr(typing, "_pulseguard_eval_shim_installed", False):
    _orig_eval_type = getattr(typing, "_eval_type", None)
    if _orig_eval_type is not None:
        def _compat_eval_type(*args, **kwargs):
            kwargs.pop("prefer_fwd_module", None)
            return _orig_eval_type(*args, **kwargs)
        typing._eval_type = _compat_eval_type
        typing._pulseguard_eval_shim_installed = True

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security, status

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Project Root & Path Setup (Always relative to project root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import platform
import re
import secrets
from backend.auth import (
    DASHBOARD_USER,
    DASHBOARD_PASSWORD_HASH,
    get_auth_status,
    get_client_ip,
    get_current_api_key,
    hash_password,
    login_rate_limiter,
    session_store,
    user_store,
    verify_api_key,
    verify_password,
)
from backend.explain import (
    assess_risk_level,
    explain_signal,
    generate_account_narrative,
    get_all_signal_definitions,
)
from backend.upload import (
    UploadValidationError,
    UploadedDataset,
    UPLOAD_TTL_MINUTES,
    generate_sample_csv,
    process_uploaded_csv,
    upload_manager,
)

DATA_DIR = Path(os.getenv("DATA_DIR", str(REPO_ROOT / "data")))


ACCOUNTS_CSV = DATA_DIR / "accounts.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
SCORES_CSV = DATA_DIR / "scores.csv"

DEFAULT_RISK_THRESHOLD = float(os.getenv("DEFAULT_RISK_THRESHOLD", "0.20"))


# ---------------------------------------------------------------------------
# In-Memory Data Store & Loader
# ---------------------------------------------------------------------------
class DataRepository:
    def __init__(self):
        self.accounts_df: Optional[pd.DataFrame] = None
        self.transactions_df: Optional[pd.DataFrame] = None
        self.scores_df: Optional[pd.DataFrame] = None
        self.merged_accounts_df: Optional[pd.DataFrame] = None

        # Fast lookup indexes
        self.account_map: Dict[str, Dict[str, Any]] = {}
        self.score_map: Dict[str, Dict[str, Any]] = {}
        self.account_sent_tx_map: Dict[str, List[int]] = {}
        self.account_recv_tx_map: Dict[str, List[int]] = {}

        # Cached dataset-wide metrics
        self.dataset_summary_cache: Optional[Dict[str, Any]] = None
        self.load_error: Optional[str] = None
        self.loaded_at: Optional[str] = None

    def verify_files_exist(self) -> List[str]:
        missing = []
        for p in (ACCOUNTS_CSV, TRANSACTIONS_CSV, SCORES_CSV):
            if not p.is_file():
                missing.append(str(p))
        return missing

    def load_data(self) -> None:
        missing = self.verify_files_exist()
        if missing:
            err = f"Required CSV files missing: {', '.join(missing)}. Please verify Phase 1 and Phase 2 data files."
            self.load_error = err
            return

        try:
            # 1. Accounts
            accts = pd.read_csv(ACCOUNTS_CSV)
            accts["account_id"] = accts["account_id"].astype(str)
            accts["account_type"] = accts["account_type"].astype(str)
            accts["is_fraud_account"] = accts["is_fraud_account"].astype(int)

            # 2. Scores
            scores = pd.read_csv(SCORES_CSV)
            scores["account_id"] = scores["account_id"].astype(str)
            scores["tgnn_score"] = scores["tgnn_score"].astype(float)
            scores["rule_flagged"] = scores["rule_flagged"].astype(int)
            scores["true_label"] = scores["true_label"].astype(int)
            for c in ("top_signal_1", "top_signal_2", "top_signal_3"):
                scores[c] = scores[c].fillna("none").astype(str)

            # 3. Transactions
            txns = pd.read_csv(TRANSACTIONS_CSV)
            txns["transaction_id"] = txns["transaction_id"].astype(str)
            txns["timestamp"] = txns["timestamp"].astype(str)
            txns["sender_id"] = txns["sender_id"].astype(str)
            txns["receiver_id"] = txns["receiver_id"].astype(str)
            txns["amount"] = txns["amount"].astype(float)
            txns["tx_type"] = txns["tx_type"].astype(str)
            txns["is_fraud"] = txns["is_fraud"].astype(int)

            # Merge accounts and scores
            merged = pd.merge(accts, scores, on="account_id", how="left")
            merged["tgnn_score"] = merged["tgnn_score"].fillna(0.0)
            merged["rule_flagged"] = merged["rule_flagged"].fillna(0).astype(int)
            merged["true_label"] = merged["true_label"].fillna(0).astype(int)
            merged["top_signal_1"] = merged["top_signal_1"].fillna("none")
            merged["top_signal_2"] = merged["top_signal_2"].fillna("none")
            merged["top_signal_3"] = merged["top_signal_3"].fillna("none")
            merged.sort_values(by="tgnn_score", ascending=False, inplace=True)

            self.accounts_df = accts
            self.scores_df = scores
            self.transactions_df = txns
            self.merged_accounts_df = merged

            # Build fast hash map indexes
            self.account_map = {row["account_id"]: row for row in accts.to_dict(orient="records")}
            self.score_map = {row["account_id"]: row for row in scores.to_dict(orient="records")}

            # Build transaction indices for fast account history queries
            sent_map: Dict[str, List[int]] = {}
            recv_map: Dict[str, List[int]] = {}
            for idx, row in enumerate(txns.itertuples(index=False)):
                s_id = row.sender_id
                r_id = row.receiver_id
                sent_map.setdefault(s_id, []).append(idx)
                recv_map.setdefault(r_id, []).append(idx)

            self.account_sent_tx_map = sent_map
            self.account_recv_tx_map = recv_map

            # Pre-compute static aggregate metrics
            total_tx = len(txns)
            total_vol = float(txns["amount"].sum())
            fraud_tx_mask = txns["is_fraud"] == 1
            fraud_tx_count = int(fraud_tx_mask.sum())
            fraud_vol = float(txns.loc[fraud_tx_mask, "amount"].sum())

            self.dataset_summary_cache = {
                "total_accounts": len(accts),
                "total_transactions": total_tx,
                "total_volume": round(total_vol, 2),
                "average_transaction_amount": round(float(txns["amount"].mean()), 2) if total_tx > 0 else 0.0,
                "fraud_transactions_count": fraud_tx_count,
                "fraud_transaction_pct": round((fraud_tx_count / total_tx * 100), 2) if total_tx > 0 else 0.0,
                "fraud_volume": round(fraud_vol, 2),
                "fraud_volume_pct": round((fraud_vol / total_vol * 100), 2) if total_vol > 0 else 0.0,
                "simulation_window": {
                    "start": txns["timestamp"].min() if total_tx > 0 else None,
                    "end": txns["timestamp"].max() if total_tx > 0 else None,
                },
                "account_types": accts["account_type"].value_counts().to_dict(),
                "ground_truth_fraud_accounts": int(accts["is_fraud_account"].sum()),
            }

            self.load_error = None
            self.loaded_at = datetime.utcnow().isoformat() + "Z"

        except Exception as exc:
            self.load_error = f"Error reading or parsing CSV files: {str(exc)}"

    def ensure_ready(self) -> None:
        if self.accounts_df is None and self.load_error is None:
            self.load_data()
        if self.load_error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Backend data not ready: {self.load_error}",
            )
        if self.accounts_df is None or self.transactions_df is None or self.scores_df is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Backend data has not been initialized. Please ensure CSV files are present.",
            )


repo = DataRepository()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load CSV data relative to project root
    repo.load_data()
    yield
    # Shutdown


# ---------------------------------------------------------------------------
# FastAPI Application Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PulseGuard-Z API",
    description=(
        "Operational REST backend for TGNN-based G2P welfare flash fraud detection on IBM LinuxONE.\n\n"
        "**Key Features:**\n"
        "- Real-time dashboard summaries computed from simulation datasets\n"
        "- Ranked suspicious account queues with configurable risk thresholds\n"
        "- Multi-signal explainability mapping graph features to AML typologies\n"
        "- Paginated, filterable transactional queries with sub-millisecond response times\n\n"
        "**Data Fidelity Notice:** All data served is synthetic benchmark data developed for the "
        "IBM Z Hackathon (event code SSF2026)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware: restrict to explicit allowed origins via environment
DEFAULT_CORS_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
cors_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if cors_env:
    allowed_origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
else:
    allowed_origins = DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Response Schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    phase: str
    data_loaded: bool
    records_count: Dict[str, int]
    default_risk_threshold: float
    data_paths: Dict[str, str]
    auth: Dict[str, Any]
    notice: str


class DashboardSummaryResponse(BaseModel):
    dataset_summary: Dict[str, Any]
    risk_summary: Dict[str, Any]
    account_type_distribution: Dict[str, int]
    transaction_type_distribution: List[Dict[str, Any]]
    top_risk_signals_frequency: Dict[str, int]
    threshold_applied: float
    data_fidelity_notice: str


class TransactionItem(BaseModel):
    transaction_id: str
    timestamp: str
    sender_id: str
    receiver_id: str
    amount: float
    tx_type: str
    is_fraud: int


class TransactionsPageResponse(BaseModel):
    page: int
    page_size: int
    total_records: int
    total_pages: int
    has_next: bool
    has_prev: bool
    filters_applied: Dict[str, Any]
    transactions: List[TransactionItem]


class SuspiciousAccountItem(BaseModel):
    account_id: str
    account_type: str
    tgnn_score: float
    rule_flagged: int
    true_label: int
    is_fraud_account: int
    risk_level: str
    top_signal_1: str
    top_signal_2: str
    top_signal_3: str
    signals_summary: List[str]


class SuspiciousAccountsResponse(BaseModel):
    threshold_applied: float
    total_flagged: int
    page: int
    page_size: int
    total_pages: int
    accounts: List[SuspiciousAccountItem]
    risk_interpretation_note: str


class AccountDetailResponse(BaseModel):
    account_id: str
    account_type: str
    is_fraud_account: int
    scoring: Dict[str, Any]
    statistics: Dict[str, Any]
    investigation: Dict[str, Any]
    recent_transactions: List[TransactionItem]


class GraphNodeItem(BaseModel):
    id: str
    label: str
    account_type: str
    tgnn_score: float
    is_flagged: bool
    rule_flagged: int
    is_fraud_account: int
    risk_level: str
    top_signals: List[str]


class GraphEdgeItem(BaseModel):
    id: str
    source: str
    target: str
    amount: float
    tx_type: str
    is_fraud: int
    timestamp: str


class GraphResponse(BaseModel):
    nodes: List[GraphNodeItem]
    edges: List[GraphEdgeItem]
    nodes_count: int
    edges_count: int
    flagged_nodes_count: int
    fraud_edges_count: int
    threshold: float
    focus_account_id: Optional[str]
    data_notice: str


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="Operator username")
    password: str = Field(..., min_length=8, max_length=128, description="Operator password")


class RegisterResponse(BaseModel):
    status: str
    message: str
    username: str
    role: str = "operator"
    created_at: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str = "operator"
    expires_at: str
    token_type: str = "Bearer"


class LogoutResponse(BaseModel):
    status: str
    message: str


class SessionCheckResponse(BaseModel):
    valid: bool
    username: Optional[str] = None
    role: Optional[str] = None
    expires_at: Optional[str] = None
    detail: Optional[str] = None


class SystemInfoResponse(BaseModel):
    machine: str
    system: str
    processor: str
    python_version: str
    platform_string: str
    service: str = "PulseGuard-Z"
    status: str = "online"


class UploadResponse(BaseModel):
    status: str
    dataset_name: str
    total_accounts: int
    total_transactions: int
    total_volume: float
    average_transaction_amount: float
    simulation_window: Dict[str, Optional[str]]
    flagged_accounts_at_threshold: int
    threshold_used: float
    expires_in_minutes: int
    notice: str


class UploadStatusResponse(BaseModel):
    is_uploaded: bool
    dataset_name: Optional[str] = None
    total_accounts: Optional[int] = None
    total_transactions: Optional[int] = None
    total_volume: Optional[float] = None
    expires_at: Optional[str] = None
    seconds_remaining: Optional[int] = None
    active_mode: str
    notice: str


def resolve_session_id(request: Request, credential: Optional[str] = None) -> str:
    """
    Safely resolves a unique session isolation identifier for per-user upload segregation.
    Ensures that uploaded datasets and metrics never leak between different operators or judges.
    """
    # 1. Bearer session token
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return f"bearer:{token}"

    # 2. X-API-Key header
    api_key_header = request.headers.get("x-api-key", "").strip()
    if api_key_header:
        return f"apikey:{api_key_header}"

    # 3. Explicit client session tracking header (for dev mode isolation between browser tabs)
    client_sid = request.headers.get("x-session-id", "").strip()
    if client_sid:
        return f"client:{client_sid}"

    # 4. Auth credential string if provided by verify_api_key
    if credential and not credential.startswith("dev-mode"):
        return f"cred:{credential}"

    # 5. Client IP fallback for open dev mode
    client_ip = get_client_ip(request)
    return f"ip:{client_ip}"


# ---------------------------------------------------------------------------
# Endpoints Implementation
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check & dataset status",
    tags=["System"],
)
def get_health():
    """
    Returns system operational health, CSV loading status, record counts,
    and active security configuration.
    """
    if repo.accounts_df is None and repo.load_error is None:
        repo.load_data()

    records = {
        "accounts": len(repo.accounts_df) if repo.accounts_df is not None else 0,
        "transactions": len(repo.transactions_df) if repo.transactions_df is not None else 0,
        "scores": len(repo.scores_df) if repo.scores_df is not None else 0,
    }
    is_loaded = repo.load_error is None and repo.accounts_df is not None


    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        service="PulseGuard-Z Backend API",
        version="1.0.0",
        phase="Phase 3: Operational API & Explainability",
        data_loaded=is_loaded,
        records_count=records,
        default_risk_threshold=DEFAULT_RISK_THRESHOLD,
        data_paths={
            "accounts_csv": str(ACCOUNTS_CSV),
            "transactions_csv": str(TRANSACTIONS_CSV),
            "scores_csv": str(SCORES_CSV),
        },
        auth={**get_auth_status(), "active_uploads": upload_manager.active_upload_count()},
        notice=(
            "Benchmark data simulating G2P welfare flash fraud on IBM LinuxONE (SSF2026). "
            "All metrics computed strictly from local CSV datasets."
        ),
    )


@router.get(
    "/auth/status",
    summary="Authentication configuration status",
    tags=["System"],
)
def auth_status():
    """
    Inspects whether API key verification is enforced or operating in open development mode.
    """
    return get_auth_status()


@router.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register new operator account",
    tags=["Authentication"],
)
def register(req: RegisterRequest, request: Request):
    """
    Registers a new operator user account with cryptographically secure PBKDF2-HMAC-SHA256 password hashing.
    Server-side validation guarantees:
    - Minimum 3 and maximum 32 alphanumeric/dash/underscore/dot characters for username.
    - Password must be at least 8 characters and contain at least one letter and one number or special character.
    - Administrator role is NEVER granted to newly registered users (strictly granted 'operator').
    - Usernames matching the configured administrator or system keywords are strictly reserved.
    - Prevents duplicate username registrations.
    """
    client_ip = get_client_ip(request)

    if login_rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please try again after 5 minutes.",
        )

    clean_u = req.username.strip()
    if not re.match(r"^[a-zA-Z0-9_\-\.]{3,32}$", clean_u):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username must be between 3 and 32 characters and contain only letters, numbers, underscores, dashes, or dots.",
        )

    # Reserved username check - never allow registering or impersonating administrator
    reserved_names = {"admin", "administrator", "root", "system", "pulseguard", "supervisor"}
    if DASHBOARD_USER:
        reserved_names.add(DASHBOARD_USER.lower())

    if clean_u.lower() in reserved_names:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{clean_u}' is reserved. Administrator privileges cannot be registered.",
        )

    if user_store.user_exists(clean_u):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{clean_u}' is already registered. Please choose another username.",
        )

    # Password strength validation
    if len(req.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters long.",
        )
    if len(req.password) > 128:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password cannot exceed 128 characters.",
        )
    has_letter = any(c.isalpha() for c in req.password)
    has_num_or_symbol = any(c.isdigit() or not c.isalnum() for c in req.password)
    if not (has_letter and has_num_or_symbol):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must contain at least one letter and at least one number or special character.",
        )

    pwd_hash = hash_password(req.password)
    # Strictly enforce operator role: newly registered users NEVER receive admin privileges
    user_record = user_store.create_user(username=clean_u, password_hash=pwd_hash, role="operator")

    return RegisterResponse(
        status="success",
        message="Operator account registered successfully. You can now sign in.",
        username=user_record["username"],
        role="operator",
        created_at=user_record["created_at"],
    )


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    summary="Dashboard user authentication",
    tags=["Authentication"],
)
def login(req: LoginRequest, request: Request):
    """
    Authenticates a dashboard operator or configured administrator.
    Supports:
    1. Static administrator configured via DASHBOARD_USER and DASHBOARD_PASSWORD_HASH (admin role)
    2. Registered operators stored in user_store (operator role)
    Applies IP rate limiting: max 5 failed attempts per IP within a 5-minute rolling window.
    On success, returns an unguessable 32-byte in-memory Bearer session token.
    """
    client_ip = get_client_ip(request)

    if login_rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again after 5 minutes.",
        )

    clean_u = req.username.strip()
    auth_success = False
    auth_role = "operator"
    auth_username = clean_u

    # 1. Check configured administrator credentials
    if DASHBOARD_USER and secrets.compare_digest(clean_u.lower(), DASHBOARD_USER.lower()):
        if DASHBOARD_PASSWORD_HASH and verify_password(req.password, DASHBOARD_PASSWORD_HASH):
            auth_success = True
            auth_role = "admin"
            auth_username = DASHBOARD_USER

    # 2. Check registered user accounts if not authenticated as admin
    if not auth_success:
        user_record = user_store.get_user(clean_u)
        if user_record and verify_password(req.password, user_record["password_hash"]):
            auth_success = True
            auth_role = user_record.get("role", "operator")
            auth_username = user_record["username"]

    if not auth_success:
        login_rate_limiter.record_failure(client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    login_rate_limiter.record_success(client_ip)
    token, expires_at = session_store.create_session(username=auth_username, role=auth_role)
    return LoginResponse(
        token=token,
        username=auth_username,
        role=auth_role,
        expires_at=expires_at.isoformat(),
        token_type="Bearer",
    )


@router.post(
    "/auth/logout",
    response_model=LogoutResponse,
    summary="Revoke active session token",
    tags=["Authentication"],
)
def logout(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
    token: Optional[str] = Security(get_current_api_key),
):
    """
    Revokes the current Bearer session token from memory.
    """
    if token:
        session_store.revoke_session(token)
    return LogoutResponse(status="success", message="Session successfully logged out.")


@router.get(
    "/auth/session",
    response_model=SessionCheckResponse,
    summary="Validate active session token",
    tags=["Authentication"],
)
def check_session(
    token: Optional[str] = Security(get_current_api_key),
):
    """
    Inspects whether the provided Bearer token corresponds to an active, unexpired session.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No session token provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    session = session_store.get_session(token)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return SessionCheckResponse(
        valid=True,
        username=session["username"],
        role=session.get("role", "operator"),
        expires_at=session["expires_at"].isoformat(),
    )


@router.get(
    "/system",
    response_model=SystemInfoResponse,
    summary="System and platform architecture information",
    tags=["System"],
)
def get_system_info():
    """
    Returns runtime host system information, architecture (using platform.machine()),
    and platform environment metadata.
    """
    machine_arch = platform.machine() or "unknown"
    sys_name = platform.system() or "unknown"
    proc = platform.processor() or machine_arch

    return SystemInfoResponse(
        machine=machine_arch,
        system=sys_name,
        processor=proc,
        python_version=platform.python_version(),
        platform_string=f"{sys_name} ({machine_arch})",
        service="PulseGuard-Z",
        status="online",
    )


# ---------------------------------------------------------------------------
# CSV Upload & Dataset Session Management
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload transaction CSV for in-memory fraud risk analysis",
    tags=["Upload"],
)
async def upload_csv(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Accepts arbitrary transaction CSV data (up to 10 MB and 200,000 rows) as raw text/csv body.
    Processes 100% in memory; strictly isolates dataset to the authenticated session for 30 minutes.
    """
    session_id = resolve_session_id(request, api_key)

    body_bytes = await request.body()
    if not body_bytes or len(body_bytes.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload body. Please send CSV data in the request body with Content-Type: text/csv.",
        )

    filename = "uploaded_transactions.csv"
    cd = request.headers.get("content-disposition", "")
    if "filename=" in cd:
        parts = cd.split("filename=")
        if len(parts) > 1:
            raw_fn = parts[1].split(";")[0].strip('"\' ')
            if raw_fn:
                filename = raw_fn

    try:
        dataset = process_uploaded_csv(session_id, body_bytes, filename=filename)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV Validation Error [{exc.code}]: {exc.message}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process CSV upload: {str(exc)}",
        )

    flagged_count = int(((dataset.merged_accounts_df["tgnn_score"] >= DEFAULT_RISK_THRESHOLD) | (dataset.merged_accounts_df["rule_flagged"] == 1)).sum())

    return UploadResponse(
        status="success",
        dataset_name=filename,
        total_accounts=dataset.summary["total_accounts"],
        total_transactions=dataset.summary["total_transactions"],
        total_volume=dataset.summary["total_volume"],
        average_transaction_amount=dataset.summary["average_transaction_amount"],
        simulation_window=dataset.summary["simulation_window"],
        flagged_accounts_at_threshold=flagged_count,
        threshold_used=DEFAULT_RISK_THRESHOLD,
        expires_in_minutes=UPLOAD_TTL_MINUTES,
        notice=dataset.summary["notice"],
    )


@router.get(
    "/upload/sample",
    summary="Download sample compliant CSV template",
    tags=["Upload"],
)
def download_sample_csv():
    """
    Returns a downloadable sample CSV compliant with all upload validation criteria.
    """
    sample_text = generate_sample_csv()
    return PlainTextResponse(
        content=sample_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="pulseguard_sample_transactions.csv"',
        },
    )


@router.post(
    "/upload/reset",
    summary="Clear uploaded dataset and return to default synthetic demo",
    tags=["Upload"],
)
def reset_to_demo(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Purges any custom uploaded dataset for the calling session, instantly reverting
    the dashboard view to the default shared synthetic G2P demo without restarting.
    """
    session_id = resolve_session_id(request, api_key)
    removed = upload_manager.remove_dataset(session_id)
    return {
        "status": "success",
        "message": "Uploaded dataset cleared. Dashboard has returned to the default synthetic demo.",
        "was_active": removed,
    }


@router.get(
    "/upload/status",
    response_model=UploadStatusResponse,
    summary="Check current dataset status for active session",
    tags=["Upload"],
)
def get_upload_status(
    request: Request,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Returns whether the current user session is viewing a custom uploaded dataset or the default demo,
    along with time remaining before in-memory expiration.
    """
    session_id = resolve_session_id(request, api_key)
    dataset = upload_manager.get_dataset(session_id)

    if dataset:
        now = datetime.now(timezone.utc)
        rem_sec = max(0, int((dataset.expires_at - now).total_seconds()))
        return UploadStatusResponse(
            is_uploaded=True,
            dataset_name=dataset.summary.get("dataset_name"),
            total_accounts=dataset.summary.get("total_accounts"),
            total_transactions=dataset.summary.get("total_transactions"),
            total_volume=dataset.summary.get("total_volume"),
            expires_at=dataset.expires_at.isoformat(),
            seconds_remaining=rem_sec,
            active_mode="custom_upload",
            notice=dataset.summary.get("notice", ""),
        )

    return UploadStatusResponse(
        is_uploaded=False,
        dataset_name="synthetic_g2p_benchmark",
        total_accounts=len(repo.accounts_df) if repo.accounts_df is not None else 2000,
        total_transactions=len(repo.transactions_df) if repo.transactions_df is not None else 50000,
        total_volume=repo.dataset_summary_cache.get("total_volume") if repo.dataset_summary_cache else 0.0,
        expires_at=None,
        seconds_remaining=None,
        active_mode="default_demo",
        notice="Displaying default synthetic G2P demo dataset (SSF2026).",
    )


@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
    summary="Dashboard metrics & aggregate fraud statistics",
    tags=["Dashboard"],
)
def get_dashboard_summary(
    request: Request,
    threshold: float = Query(
        default=DEFAULT_RISK_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Calibrated TGNN decision threshold (default: 0.20).",
    ),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Calculates comprehensive fraud monitoring metrics from the underlying CSVs:
    - Total accounts, transaction volume, and fraud proportion
    - Risk score distribution across accounts
    - Flagged account counts according to the chosen threshold
    - Account type breakdown and transaction type volumes
    - Most frequent top signals among flagged accounts
    """
    session_id = resolve_session_id(request, api_key)
    uploaded_ds = upload_manager.get_dataset(session_id)

    if uploaded_ds:
        m_df = uploaded_ds.merged_accounts_df
        txns = uploaded_ds.transactions_df
        ds_summary = uploaded_ds.summary
        is_uploaded = True
    else:
        repo.ensure_ready()
        m_df = repo.merged_accounts_df
        txns = repo.transactions_df
        ds_summary = repo.dataset_summary_cache
        is_uploaded = False

    # Evaluate against threshold
    tgnn_mask = m_df["tgnn_score"] >= threshold
    rule_mask = m_df["rule_flagged"] == 1
    suspicious_mask = tgnn_mask | rule_mask
    dual_mask = tgnn_mask & rule_mask
    tgnn_only_mask = tgnn_mask & (~rule_mask)
    rule_only_mask = rule_mask & (~tgnn_mask)

    scores_series = m_df["tgnn_score"]

    # Score distribution buckets
    bucket_defs = [
        ("0.00 - 0.05", (scores_series >= 0.00) & (scores_series < 0.05)),
        ("0.05 - 0.10", (scores_series >= 0.05) & (scores_series < 0.10)),
        ("0.10 - 0.15", (scores_series >= 0.10) & (scores_series < 0.15)),
        ("0.15 - 0.20", (scores_series >= 0.15) & (scores_series < 0.20)),
        ("0.20 - 0.25", (scores_series >= 0.20) & (scores_series < 0.25)),
        ("0.25 - 0.30", (scores_series >= 0.25)),
    ]
    distribution_buckets = {label: int(mask.sum()) for label, mask in bucket_defs}

    # Top signals frequency among suspicious accounts
    suspicious_df = m_df[suspicious_mask]
    signal_counts: Dict[str, int] = {}
    for col in ("top_signal_1", "top_signal_2", "top_signal_3"):
        for sig, count in suspicious_df[col].value_counts().items():
            if sig and sig != "none":
                signal_counts[sig] = signal_counts.get(sig, 0) + int(count)

    sorted_signals = dict(sorted(signal_counts.items(), key=lambda x: x[1], reverse=True))

    # Transaction types breakdown
    tx_type_group = txns.groupby("tx_type").agg(
        count=("transaction_id", "count"),
        total_volume=("amount", "sum"),
        fraud_count=("is_fraud", "sum"),
    ).reset_index()

    tx_type_list = [
        {
            "tx_type": row["tx_type"],
            "count": int(row["count"]),
            "total_volume": round(float(row["total_volume"]), 2),
            "fraud_count": int(row["fraud_count"]),
            "fraud_rate_pct": round(float(row["fraud_count"]) / float(row["count"]) * 100, 2) if row["count"] > 0 else 0.0,
        }
        for row in tx_type_group.to_dict(orient="records")
    ]

    risk_summary = {
        "threshold_applied": threshold,
        "suspicious_accounts_count": int(suspicious_mask.sum()),
        "rule_flagged_accounts_count": int(rule_mask.sum()),
        "dual_flagged_count": int(dual_mask.sum()),
        "tgnn_only_flagged_count": int(tgnn_only_mask.sum()),
        "rule_only_flagged_count": int(rule_only_mask.sum()),
        "score_statistics": {
            "min": round(float(scores_series.min()), 4) if len(scores_series) > 0 else 0.0,
            "max": round(float(scores_series.max()), 4) if len(scores_series) > 0 else 0.0,
            "mean": round(float(scores_series.mean()), 4) if len(scores_series) > 0 else 0.0,
            "median": round(float(scores_series.median()), 4) if len(scores_series) > 0 else 0.0,
            "std": round(float(scores_series.std()), 4) if len(scores_series) > 1 else 0.0,
        },
        "score_distribution_buckets": distribution_buckets,
    }

    fidelity_notice = (
        "Custom uploaded dataset analyzed in-memory. Risk scores represent exploratory "
        "topological estimates from TGNN trained on synthetic G2P data; estimates have not been "
        "validated on real-world transaction data, and no ground-truth fraud labels exist."
        if is_uploaded
        else "Metrics computed directly from local synthetic G2P simulation CSV data (SSF2026)."
    )

    return DashboardSummaryResponse(
        dataset_summary=ds_summary,
        risk_summary=risk_summary,
        account_type_distribution=ds_summary.get("account_types", {}),
        transaction_type_distribution=tx_type_list,
        top_risk_signals_frequency=sorted_signals,
        threshold_applied=threshold,
        data_fidelity_notice=fidelity_notice,
    )


@router.get(
    "/transactions",
    response_model=TransactionsPageResponse,
    summary="Paginated and filterable transaction search",
    tags=["Transactions"],
)
def get_transactions(
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=50, ge=1, le=1000, description="Records per page (max 1000)."),
    account_id: Optional[str] = Query(
        default=None, description="Filter for transactions where account is sender OR receiver."
    ),
    sender_id: Optional[str] = Query(default=None, description="Filter by sender account ID."),
    receiver_id: Optional[str] = Query(default=None, description="Filter by receiver account ID."),
    tx_type: Optional[str] = Query(
        default=None,
        description="Filter by transaction type (retail_purchase, welfare_payout, p2p_transfer, agent_cash_out, utility_payment).",
    ),
    is_fraud: Optional[int] = Query(
        default=None, ge=0, le=1, description="Filter by ground-truth fraud flag (0 or 1)."
    ),
    min_amount: Optional[float] = Query(default=None, ge=0.0, description="Minimum transaction amount."),
    max_amount: Optional[float] = Query(default=None, ge=0.0, description="Maximum transaction amount."),
    start_time: Optional[str] = Query(default=None, description="Filter timestamp >= start_time (YYYY-MM-DD HH:MM:SS)."),
    end_time: Optional[str] = Query(default=None, description="Filter timestamp <= end_time (YYYY-MM-DD HH:MM:SS)."),
    sort_by: str = Query(
        default="timestamp",
        pattern="^(timestamp|amount|transaction_id)$",
        description="Column to sort by.",
    ),
    sort_order: str = Query(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort direction (asc or desc).",
    ),

    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Queries and paginates through transactions with multi-field filtering.
    """
    session_id = resolve_session_id(request, api_key)
    uploaded_ds = upload_manager.get_dataset(session_id)
    if uploaded_ds:
        txns = uploaded_ds.transactions_df
    else:
        repo.ensure_ready()
        txns = repo.transactions_df

    # Quick indexed path if single account_id is requested and no other complex time filter is given
    mask = pd.Series(True, index=txns.index)

    if account_id:
        aid = account_id.strip()
        mask = mask & ((txns["sender_id"] == aid) | (txns["receiver_id"] == aid))

    if sender_id:
        mask = mask & (txns["sender_id"] == sender_id.strip())

    if receiver_id:
        mask = mask & (txns["receiver_id"] == receiver_id.strip())

    if tx_type:
        mask = mask & (txns["tx_type"] == tx_type.strip().lower())

    if is_fraud is not None:
        mask = mask & (txns["is_fraud"] == is_fraud)

    if min_amount is not None:
        mask = mask & (txns["amount"] >= min_amount)

    if max_amount is not None:
        mask = mask & (txns["amount"] <= max_amount)

    if start_time:
        mask = mask & (txns["timestamp"] >= start_time.strip())

    if end_time:
        mask = mask & (txns["timestamp"] <= end_time.strip())

    filtered_df = txns[mask]
    total_records = len(filtered_df)
    total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1

    if total_records == 0 or page > total_pages:
        paged_records = []
    else:
        # Sort
        ascending = sort_order.lower() == "asc"
        sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        slice_df = sorted_df.iloc[start_idx:end_idx]
        paged_records = [
            TransactionItem(
                transaction_id=row["transaction_id"],
                timestamp=row["timestamp"],
                sender_id=row["sender_id"],
                receiver_id=row["receiver_id"],
                amount=float(row["amount"]),
                tx_type=row["tx_type"],
                is_fraud=int(row["is_fraud"]),
            )
            for row in slice_df.to_dict(orient="records")
        ]

    filters_applied = {
        k: v
        for k, v in {
            "account_id": account_id,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "tx_type": tx_type,
            "is_fraud": is_fraud,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "start_time": start_time,
            "end_time": end_time,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }.items()
        if v is not None
    }

    return TransactionsPageResponse(
        page=page,
        page_size=page_size,
        total_records=total_records,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
        filters_applied=filters_applied,
        transactions=paged_records,
    )


@router.get(
    "/accounts/suspicious",
    response_model=SuspiciousAccountsResponse,
    summary="Suspicious accounts priority queue",
    tags=["Accounts"],
)
def get_suspicious_accounts(
    request: Request,
    threshold: float = Query(
        default=DEFAULT_RISK_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Risk score cutoff (default: 0.20, calibrated for class imbalance).",
    ),
    rule_flagged_only: bool = Query(
        default=False,
        description="Filter exclusively for accounts flagged by static baseline rules.",
    ),
    account_type: Optional[str] = Query(
        default=None,
        description="Filter by account type: beneficiary, merchant, agent, government.",
    ),
    sort_by: str = Query(
        default="tgnn_score",
        pattern="^(tgnn_score|account_id)$",
        description="Sorting key.",
    ),
    sort_order: str = Query(
        default="desc",
        pattern="^(asc|desc)$",
        description="Sort direction.",
    ),

    page: int = Query(default=1, ge=1, description="Page number."),
    page_size: int = Query(default=50, ge=1, le=500, description="Items per page."),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Returns accounts prioritized for fraud compliance review based on the TGNN graph risk score.
    """
    session_id = resolve_session_id(request, api_key)
    uploaded_ds = upload_manager.get_dataset(session_id)
    if uploaded_ds:
        df = uploaded_ds.merged_accounts_df
        is_uploaded = True
    else:
        repo.ensure_ready()
        df = repo.merged_accounts_df
        is_uploaded = False

    # Mask construction
    tgnn_mask = df["tgnn_score"] >= threshold
    rule_mask = df["rule_flagged"] == 1
    mask = rule_mask if rule_flagged_only else (tgnn_mask | rule_mask)

    if account_type:
        mask = mask & (df["account_type"] == account_type.strip().lower())

    filtered_df = df[mask]
    total_flagged = len(filtered_df)
    total_pages = math.ceil(total_flagged / page_size) if total_flagged > 0 else 1

    if total_flagged == 0 or page > total_pages:
        accounts_list = []
    else:
        ascending = sort_order.lower() == "asc"
        sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        slice_df = sorted_df.iloc[start_idx:end_idx]

        accounts_list = []
        for row in slice_df.to_dict(orient="records"):
            sigs = [row["top_signal_1"], row["top_signal_2"], row["top_signal_3"]]
            clean_sigs = [s for s in sigs if s and s != "none"]
            sig_summaries = [explain_signal(s)["title"] for s in clean_sigs]
            risk_tier = assess_risk_level(row["tgnn_score"], threshold, bool(row["rule_flagged"]))

            accounts_list.append(
                SuspiciousAccountItem(
                    account_id=row["account_id"],
                    account_type=row["account_type"],
                    tgnn_score=round(float(row["tgnn_score"]), 4),
                    rule_flagged=int(row["rule_flagged"]),
                    true_label=int(row["true_label"]),
                    is_fraud_account=int(row["is_fraud_account"]),
                    risk_level=risk_tier,
                    top_signal_1=row["top_signal_1"],
                    top_signal_2=row["top_signal_2"],
                    top_signal_3=row["top_signal_3"],
                    signals_summary=sig_summaries,
                )
            )

    risk_note = (
        "Scores represent exploratory topological graph risk from TGNN embeddings on uploaded data. "
        "Flagged entities are prioritized for compliance review. No ground-truth labels exist."
        if is_uploaded
        else "Scores represent topological graph risk from TGNN embeddings, not absolute probabilities. "
             "Flagged entities are prioritized for human-in-the-loop compliance review."
    )

    return SuspiciousAccountsResponse(
        threshold_applied=threshold,
        total_flagged=total_flagged,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        accounts=accounts_list,
        risk_interpretation_note=risk_note,
    )


@router.get(
    "/accounts/{account_id}",
    response_model=AccountDetailResponse,
    summary="Comprehensive account profile & risk evaluation",
    tags=["Accounts"],
)
def get_account_detail(
    account_id: str,
    request: Request,
    threshold: float = Query(
        default=DEFAULT_RISK_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Calibrated decision threshold to evaluate suspicion against.",
    ),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Retrieves full profile for an account, including graph risk score, rule flags,
    historical transaction volumes, top driving signals, and recent activity.
    """
    session_id = resolve_session_id(request, api_key)
    uploaded_ds = upload_manager.get_dataset(session_id)
    aid = account_id.strip()

    if uploaded_ds:
        if aid not in uploaded_ds.account_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{aid}' not found in uploaded dataset.",
            )
        acct_data = uploaded_ds.account_map[aid]
        score_data = uploaded_ds.score_map.get(aid, {
            "tgnn_score": 0.0,
            "rule_flagged": 0,
            "true_label": 0,
            "top_signal_1": "none",
            "top_signal_2": "none",
            "top_signal_3": "none",
        })
        txns = uploaded_ds.transactions_df
        sent_indices = uploaded_ds.account_sent_tx_map.get(aid, [])
        recv_indices = uploaded_ds.account_recv_tx_map.get(aid, [])
    else:
        repo.ensure_ready()
        if aid not in repo.account_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{aid}' not found in dataset.",
            )
        acct_data = repo.account_map[aid]
        score_data = repo.score_map.get(aid, {
            "tgnn_score": 0.0,
            "rule_flagged": 0,
            "true_label": 0,
            "top_signal_1": "none",
            "top_signal_2": "none",
            "top_signal_3": "none",
        })
        txns = repo.transactions_df
        sent_indices = repo.account_sent_tx_map.get(aid, [])
        recv_indices = repo.account_recv_tx_map.get(aid, [])

    tgnn_score = float(score_data["tgnn_score"])
    rule_flagged = bool(score_data["rule_flagged"])
    signals = [
        score_data.get("top_signal_1", "none"),
        score_data.get("top_signal_2", "none"),
        score_data.get("top_signal_3", "none"),
    ]

    # Aggregate transaction statistics using pre-computed index
    sent_df = txns.iloc[sent_indices] if sent_indices else pd.DataFrame(columns=txns.columns)
    recv_df = txns.iloc[recv_indices] if recv_indices else pd.DataFrame(columns=txns.columns)

    sent_count = len(sent_df)
    sent_amount = float(sent_df["amount"].sum()) if sent_count > 0 else 0.0
    sent_avg = round(sent_amount / sent_count, 2) if sent_count > 0 else 0.0

    recv_count = len(recv_df)
    recv_amount = float(recv_df["amount"].sum()) if recv_count > 0 else 0.0
    recv_avg = round(recv_amount / recv_count, 2) if recv_count > 0 else 0.0

    fraud_sent = int(sent_df["is_fraud"].sum()) if sent_count > 0 else 0
    fraud_recv = int(recv_df["is_fraud"].sum()) if recv_count > 0 else 0

    unique_senders = int(recv_df["sender_id"].nunique()) if recv_count > 0 else 0
    unique_receivers = int(sent_df["receiver_id"].nunique()) if sent_count > 0 else 0

    all_indices = sent_indices + recv_indices
    if all_indices:
        all_tx_df = txns.iloc[all_indices].sort_values(by="timestamp", ascending=False)
        first_act = all_tx_df["timestamp"].min()
        last_act = all_tx_df["timestamp"].max()
        recent_txs = [
            TransactionItem(
                transaction_id=r["transaction_id"],
                timestamp=r["timestamp"],
                sender_id=r["sender_id"],
                receiver_id=r["receiver_id"],
                amount=float(r["amount"]),
                tx_type=r["tx_type"],
                is_fraud=int(r["is_fraud"]),
            )
            for r in all_tx_df.head(10).to_dict(orient="records")
        ]
    else:
        first_act = None
        last_act = None
        recent_txs = []

    # Generate investigative narrative
    investigation = generate_account_narrative(
        account_id=aid,
        tgnn_score=tgnn_score,
        rule_flagged=rule_flagged,
        top_signals=signals,
        threshold=threshold,
        account_type=acct_data["account_type"],
    )

    scoring_payload = {
        "tgnn_score": round(tgnn_score, 4),
        "rule_flagged": int(rule_flagged),
        "true_label": int(score_data.get("true_label", 0)),
        "risk_level": assess_risk_level(tgnn_score, threshold, rule_flagged),
        "is_suspicious": (tgnn_score >= threshold) or rule_flagged,
        "threshold_used": threshold,
    }

    stats_payload = {
        "total_sent_count": sent_count,
        "total_sent_amount": round(sent_amount, 2),
        "average_sent_amount": sent_avg,
        "total_received_count": recv_count,
        "total_received_amount": round(recv_amount, 2),
        "average_received_amount": recv_avg,
        "net_flow_amount": round(recv_amount - sent_amount, 2),
        "fraud_transactions_involved": fraud_sent + fraud_recv,
        "unique_senders_observed": unique_senders,
        "unique_receivers_observed": unique_receivers,
        "first_activity_timestamp": first_act,
        "last_activity_timestamp": last_act,
    }

    return AccountDetailResponse(
        account_id=aid,
        account_type=acct_data["account_type"],
        is_fraud_account=int(acct_data["is_fraud_account"]),
        scoring=scoring_payload,
        statistics=stats_payload,
        investigation=investigation,
        recent_transactions=recent_txs,
    )


@router.get(
    "/explain/signals",
    summary="Catalog of all detectable fraud signals and typologies",
    tags=["Explainability"],
)
def get_signals_catalog(api_key: Optional[str] = Depends(verify_api_key)):
    """
    Returns the comprehensive catalog of all 8 temporal graph signals extracted by the model,
    including definitions, severity, and anti-money laundering typologies.
    """
    return {
        "signals_count": len(get_all_signal_definitions()),
        "signals": get_all_signal_definitions(),
        "methodology": (
            "Signals are computed across 30-minute temporal snapshot windows, tracking in/out-degree, "
            "flow ratios, and volume distributions before being embedded via 2-layer GCN and 1-layer GRU."
        ),
    }


@router.get(
    "/explain/{account_id}",
    summary="Detailed investigator explainability brief for an account",
    tags=["Explainability"],
)
def get_account_explanation(
    account_id: str,
    request: Request,
    threshold: float = Query(
        default=DEFAULT_RISK_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Decision threshold applied to evaluate risk level.",
    ),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Provides an interpretable fraud investigator brief for a specific account,
    explaining why the TGNN assigned its score and how individual graph signals interact.
    """
    session_id = resolve_session_id(request, api_key)
    uploaded_ds = upload_manager.get_dataset(session_id)
    aid = account_id.strip()

    if uploaded_ds:
        if aid not in uploaded_ds.account_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{aid}' does not exist in uploaded dataset.",
            )
        acct_data = uploaded_ds.account_map[aid]
        score_data = uploaded_ds.score_map.get(aid, {
            "tgnn_score": 0.0,
            "rule_flagged": 0,
            "true_label": 0,
            "top_signal_1": "none",
            "top_signal_2": "none",
            "top_signal_3": "none",
        })
    else:
        repo.ensure_ready()
        if aid not in repo.account_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{aid}' does not exist in dataset.",
            )
        acct_data = repo.account_map[aid]
        score_data = repo.score_map.get(aid, {
            "tgnn_score": 0.0,
            "rule_flagged": 0,
            "true_label": 0,
            "top_signal_1": "none",
            "top_signal_2": "none",
            "top_signal_3": "none",
        })

    tgnn_score = float(score_data["tgnn_score"])
    rule_flagged = bool(score_data["rule_flagged"])
    signals = [
        score_data.get("top_signal_1", "none"),
        score_data.get("top_signal_2", "none"),
        score_data.get("top_signal_3", "none"),
    ]

    narrative = generate_account_narrative(
        account_id=aid,
        tgnn_score=tgnn_score,
        rule_flagged=rule_flagged,
        top_signals=signals,
        threshold=threshold,
        account_type=acct_data["account_type"],
    )

    return narrative


@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="Bounded transaction network graph for visualization",
    tags=["Graph"],
)
def get_graph(
    request: Request,
    threshold: float = Query(
        default=DEFAULT_RISK_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Calibrated TGNN decision threshold (default: 0.20).",
    ),
    focus_account_id: Optional[str] = Query(
        default=None,
        description="Optional account ID to center an ego-network graph around.",
    ),
    max_nodes: int = Query(
        default=50,
        ge=5,
        le=150,
        description="Maximum nodes in returned subgraph.",
    ),
    max_edges: int = Query(
        default=100,
        ge=5,
        le=300,
        description="Maximum edges in returned subgraph.",
    ),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Returns a bounded, high-fidelity transaction network graph for Cytoscape.js visualization.
    Extracts real accounts and transactions, linking high-risk nodes and their counterparties.
    """
    session_id = resolve_session_id(request, api_key)
    uploaded_ds = upload_manager.get_dataset(session_id)

    if uploaded_ds:
        txns = uploaded_ds.transactions_df
        m_df = uploaded_ds.merged_accounts_df
        account_map = uploaded_ds.account_map
        score_map = uploaded_ds.score_map
        sent_tx_map = uploaded_ds.account_sent_tx_map
        recv_tx_map = uploaded_ds.account_recv_tx_map
        data_notice = (
            "Bounded transaction network extracted from custom uploaded dataset. "
            "Risk scores represent exploratory topological estimates from TGNN."
        )
    else:
        repo.ensure_ready()
        txns = repo.transactions_df
        m_df = repo.merged_accounts_df
        account_map = repo.account_map
        score_map = repo.score_map
        sent_tx_map = repo.account_sent_tx_map
        recv_tx_map = repo.account_recv_tx_map
        data_notice = "Bounded real transaction network extracted from transactions.csv and scores.csv."

    if focus_account_id:
        aid = focus_account_id.strip()
        if aid not in account_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{aid}' does not exist in dataset.",
            )
        sent_indices = sent_tx_map.get(aid, [])
        recv_indices = recv_tx_map.get(aid, [])
        all_indices = sent_indices + recv_indices

        if all_indices:
            focus_tx_df = txns.iloc[all_indices]
            senders = focus_tx_df["sender_id"].unique().tolist()
            receivers = focus_tx_df["receiver_id"].unique().tolist()
            counterparties = list((set(senders) | set(receivers)) - {aid})

            cp_scored = []
            for cp in counterparties:
                cp_score = float(score_map.get(cp, {}).get("tgnn_score", 0.0))
                cp_has_fraud = int((focus_tx_df[(focus_tx_df["sender_id"] == cp) | (focus_tx_df["receiver_id"] == cp)]["is_fraud"] == 1).sum())
                cp_scored.append((cp, cp_has_fraud, cp_score))

            cp_scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            chosen_cps = [x[0] for x in cp_scored[: max_nodes - 1]]
            selected_ids = [aid] + chosen_cps
        else:
            selected_ids = [aid]

        selected_set = set(selected_ids)
        sub_tx_mask = txns["sender_id"].isin(selected_set) & txns["receiver_id"].isin(selected_set)
        sub_tx = (
            txns[sub_tx_mask]
            .sort_values(by=["is_fraud", "amount"], ascending=[False, False])
            .head(max_edges)
        )
    else:
        # Global view: Focus on top suspicious accounts and their direct/high-value counterparties
        suspicious_list = m_df.head(min(25, max_nodes // 2))["account_id"].tolist()
        suspicious_set = set(suspicious_list)

        # 1. Direct transactions between top suspicious accounts
        direct_mask = txns["sender_id"].isin(suspicious_set) & txns["receiver_id"].isin(suspicious_set)
        direct_tx = txns[direct_mask]

        # 2. Fraud transactions incident to suspicious accounts
        fraud_mask = (
            txns["sender_id"].isin(suspicious_set) | txns["receiver_id"].isin(suspicious_set)
        ) & (txns["is_fraud"] == 1)
        fraud_tx = txns[fraud_mask]

        # 3. High-volume normal transactions to connect legitimate counterparties (merchants, etc.)
        normal_mask = (
            txns["sender_id"].isin(suspicious_set) | txns["receiver_id"].isin(suspicious_set)
        ) & (txns["is_fraud"] == 0)
        normal_tx = txns[normal_mask].sort_values(by="amount", ascending=False).head(40)

        combined_tx = pd.concat([direct_tx, fraud_tx, normal_tx]).drop_duplicates(subset=["transaction_id"])
        sorted_tx = combined_tx.sort_values(by=["is_fraud", "amount"], ascending=[False, False]).head(max_edges)

        nodes_in_edges = set(sorted_tx["sender_id"]).union(set(sorted_tx["receiver_id"]))
        selected_set = nodes_in_edges.union(suspicious_set)

        if len(selected_set) > max_nodes:
            # Retain top suspicious plus highest-degree counterparties
            sender_counts = sorted_tx["sender_id"].value_counts()
            receiver_counts = sorted_tx["receiver_id"].value_counts()
            total_counts = (sender_counts.add(receiver_counts, fill_value=0)).to_dict()

            sorted_nodes = sorted(
                list(selected_set),
                key=lambda x: (x in suspicious_set, total_counts.get(x, 0)),
                reverse=True,
            )
            selected_set = set(sorted_nodes[:max_nodes])
            sorted_tx = sorted_tx[
                sorted_tx["sender_id"].isin(selected_set) & sorted_tx["receiver_id"].isin(selected_set)
            ]

        sub_tx = sorted_tx

    # Build node objects
    node_items = []
    flagged_nodes_count = 0
    for nid in selected_set:
        acct_row = account_map.get(nid, {})
        score_row = score_map.get(nid, {})
        t_score = float(score_row.get("tgnn_score", 0.0))
        r_flag = int(score_row.get("rule_flagged", 0))
        is_flagged = bool((t_score >= threshold) or (r_flag == 1))
        if is_flagged:
            flagged_nodes_count += 1

        sigs = [
            score_row.get("top_signal_1", "none"),
            score_row.get("top_signal_2", "none"),
            score_row.get("top_signal_3", "none"),
        ]
        clean_sigs = [s for s in sigs if s and s != "none"]

        node_items.append(
            GraphNodeItem(
                id=nid,
                label=nid,
                account_type=acct_row.get("account_type", "unknown"),
                tgnn_score=round(t_score, 4),
                is_flagged=is_flagged,
                rule_flagged=r_flag,
                is_fraud_account=int(acct_row.get("is_fraud_account", 0)),
                risk_level=assess_risk_level(t_score, threshold, bool(r_flag)),
                top_signals=clean_sigs,
            )
        )

    # Sort nodes: flagged first, then by score descending
    node_items.sort(key=lambda n: (n.is_flagged, n.tgnn_score), reverse=True)

    # Build edge objects
    edge_items = []
    fraud_edges_count = 0
    for row in sub_tx.to_dict(orient="records"):
        is_f = int(row["is_fraud"])
        if is_f == 1:
            fraud_edges_count += 1
        edge_items.append(
            GraphEdgeItem(
                id=row["transaction_id"],
                source=row["sender_id"],
                target=row["receiver_id"],
                amount=round(float(row["amount"]), 2),
                tx_type=row["tx_type"],
                is_fraud=is_f,
                timestamp=row["timestamp"],
            )
        )

    return GraphResponse(
        nodes=node_items,
        edges=edge_items,
        nodes_count=len(node_items),
        edges_count=len(edge_items),
        flagged_nodes_count=flagged_nodes_count,
        fraud_edges_count=fraud_edges_count,
        threshold=threshold,
        focus_account_id=focus_account_id,
        data_notice=data_notice,
    )


# Mount router for both `/api/...` and top-level `/...` paths
app.include_router(router, prefix="/api")
app.include_router(router, prefix="")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def serve_dashboard():
    """
    Serves the Phase 4 Cytoscape.js frontend dashboard.
    """
    index_file = REPO_ROOT / "frontend" / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return JSONResponse({"status": "PulseGuard-Z backend active. Frontend file not found."})



# Custom global exception handler for clean error reporting
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ---------------------------------------------------------------------------
# CLI Direct Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"[PulseGuard-Z] Starting server on http://{host}:{port} ...")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
