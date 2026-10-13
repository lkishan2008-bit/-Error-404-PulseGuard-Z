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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Python 3.14 pre-release compatibility shim for private typing._eval_type
if sys.version_info >= (3, 14):
    _orig_eval_type = getattr(typing, "_eval_type", None)
    if _orig_eval_type is not None:
        def _compat_eval_type(*args, **kwargs):
            kwargs.pop("prefer_fwd_module", None)
            return _orig_eval_type(*args, **kwargs)
        typing._eval_type = _compat_eval_type

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Project Root & Path Setup (Always relative to project root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.auth import get_auth_status, verify_api_key
from backend.explain import (
    assess_risk_level,
    explain_signal,
    generate_account_narrative,
    get_all_signal_definitions,
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

# CORS Middleware (permits frontend dashboards or local dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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
        auth=get_auth_status(),
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


@router.get(
    "/dashboard/summary",
    response_model=DashboardSummaryResponse,
    summary="Dashboard metrics & aggregate fraud statistics",
    tags=["Dashboard"],
)
def get_dashboard_summary(
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
    - Risk score distribution across the 2,000 accounts
    - Flagged account counts according to the chosen threshold
    - Account type breakdown and transaction type volumes
    - Most frequent top signals among flagged accounts
    """
    repo.ensure_ready()
    m_df = repo.merged_accounts_df
    txns = repo.transactions_df

    # Evaluate against threshold
    suspicious_mask = m_df["tgnn_score"] >= threshold
    rule_mask = m_df["rule_flagged"] == 1
    dual_mask = suspicious_mask & rule_mask
    tgnn_only_mask = suspicious_mask & (~rule_mask)

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
            "fraud_rate_pct": round(float(row["fraud_count"]) / float(row["count"]) * 100, 2),
        }
        for row in tx_type_group.to_dict(orient="records")
    ]

    risk_summary = {
        "threshold_applied": threshold,
        "suspicious_accounts_count": int(suspicious_mask.sum()),
        "rule_flagged_accounts_count": int(rule_mask.sum()),
        "dual_flagged_count": int(dual_mask.sum()),
        "tgnn_only_flagged_count": int(tgnn_only_mask.sum()),
        "score_statistics": {
            "min": round(float(scores_series.min()), 4),
            "max": round(float(scores_series.max()), 4),
            "mean": round(float(scores_series.mean()), 4),
            "median": round(float(scores_series.median()), 4),
            "std": round(float(scores_series.std()), 4),
        },
        "score_distribution_buckets": distribution_buckets,
    }

    return DashboardSummaryResponse(
        dataset_summary=repo.dataset_summary_cache,
        risk_summary=risk_summary,
        account_type_distribution=repo.dataset_summary_cache["account_types"],
        transaction_type_distribution=tx_type_list,
        top_risk_signals_frequency=sorted_signals,
        threshold_applied=threshold,
        data_fidelity_notice=(
            "Metrics computed directly from local synthetic G2P simulation CSV data (SSF2026)."
        ),
    )


@router.get(
    "/transactions",
    response_model=TransactionsPageResponse,
    summary="Paginated and filterable transaction search",
    tags=["Transactions"],
)
def get_transactions(
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
    Queries and paginates through the 50,000 transaction records with multi-field filtering.
    """
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
    repo.ensure_ready()
    df = repo.merged_accounts_df

    # Mask construction
    mask = df["tgnn_score"] >= threshold
    if rule_flagged_only:
        mask = mask & (df["rule_flagged"] == 1)

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

    return SuspiciousAccountsResponse(
        threshold_applied=threshold,
        total_flagged=total_flagged,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        accounts=accounts_list,
        risk_interpretation_note=(
            "Scores represent topological graph risk from TGNN embeddings, not absolute probabilities. "
            "Flagged entities are prioritized for human-in-the-loop compliance review."
        ),
    )


@router.get(
    "/accounts/{account_id}",
    response_model=AccountDetailResponse,
    summary="Comprehensive account profile & risk evaluation",
    tags=["Accounts"],
)
def get_account_detail(
    account_id: str,
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
    repo.ensure_ready()
    aid = account_id.strip()

    if aid not in repo.account_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Account '{aid}' not found in the PulseGuard-Z dataset. Valid IDs follow format ACC000000 to ACC001999.",
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

    # Aggregate transaction statistics using pre-computed index
    txns = repo.transactions_df
    sent_indices = repo.account_sent_tx_map.get(aid, [])
    recv_indices = repo.account_recv_tx_map.get(aid, [])

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
    repo.ensure_ready()
    aid = account_id.strip()

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


# Mount router for both `/api/...` and top-level `/...` paths
app.include_router(router, prefix="/api")
app.include_router(router, prefix="")


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
