"""
==============================================================================
PulseGuard-Z  -  Upload Processing & Per-Session Isolation Engine
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : backend/upload.py

Purpose
-------
Handles authenticated CSV uploads, strict data validation, in-memory parsing,
per-session dataset isolation, automatic TTL expiration, and TGNN inference
adaptation for arbitrary transaction datasets.

Security & Safety Guarantees:
- Uploads processed 100% in memory; NEVER written to disk.
- Ground-truth fraud label columns (e.g., 'is_fraud', 'fraud', 'label') are STRICTLY
  rejected during header validation to prevent accidental label leakage or simulated cheating.
- Datasets are bound exclusively to the authenticated session token; sessions cannot
  access or contaminate each other's data or the default shared demo.
- Automatic TTL expiration (30 minutes) purges uploaded datasets from memory.
- Evaluation metrics (Precision/Recall/F1/ROC-AUC) are explicitly prohibited for
  uploaded data since no legitimate ground truth exists.
==============================================================================
"""

import io
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from backend.explain import assess_risk_level, explain_signal, generate_account_narrative
from ml.train_tgnn import (
    ACCT_TYPES,
    BEHAV_FEAT_DIM,
    NODE_FEAT_DIM,
    WINDOW_MINUTES,
    N_SNAPSHOTS_SEQ,
    TGNN,
    build_snapshot_features,
    build_sparse_adj,
)

# Constants & Limits
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_UPLOAD_ROWS = 200_000            # 200,000 rows
UPLOAD_TTL_MINUTES = 30              # 30-minute session expiry

REQUIRED_COLUMNS = {"timestamp", "sender_id", "receiver_id", "amount"}
PROHIBITED_LABEL_COLUMNS = {
    "is_fraud",
    "fraud",
    "label",
    "true_label",
    "is_fraud_account",
    "target",
    "ground_truth",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "ml" / "tgnn_model.pt"


class UploadValidationError(Exception):
    """Structured exception for CSV validation failures."""
    def __init__(self, message: str, code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.message = message
        self.code = code


class UploadedDataset:
    """
    Represents an in-memory processed transaction dataset submitted by an authenticated session.
    """
    def __init__(
        self,
        session_id: str,
        transactions_df: pd.DataFrame,
        accounts_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        summary: Dict[str, Any],
        ttl_minutes: int = UPLOAD_TTL_MINUTES,
    ):
        self.session_id = session_id
        self.transactions_df = transactions_df
        self.accounts_df = accounts_df
        self.scores_df = scores_df
        self.summary = summary
        self.created_at = datetime.now(timezone.utc)
        self.expires_at = self.created_at + timedelta(minutes=ttl_minutes)

        # Build fast lookup indexes
        self.account_map: Dict[str, Dict[str, Any]] = {
            r["account_id"]: r for r in accounts_df.to_dict(orient="records")
        }
        self.score_map: Dict[str, Dict[str, Any]] = {
            r["account_id"]: r for r in scores_df.to_dict(orient="records")
        }

        # Merged accounts & scores
        merged = pd.merge(accounts_df, scores_df, on="account_id", how="left")
        merged["tgnn_score"] = merged["tgnn_score"].fillna(0.0)
        merged["rule_flagged"] = merged["rule_flagged"].fillna(0).astype(int)
        merged["true_label"] = 0  # No ground truth on uploaded data
        merged["is_fraud_account"] = 0  # No ground truth on uploaded data
        for c in ("top_signal_1", "top_signal_2", "top_signal_3"):
            merged[c] = merged[c].fillna("none").astype(str)
        merged.sort_values(by="tgnn_score", ascending=False, inplace=True)
        self.merged_accounts_df = merged

        # Transaction fast indexes
        sent_map: Dict[str, List[int]] = {}
        recv_map: Dict[str, List[int]] = {}
        for idx, row in enumerate(transactions_df.itertuples(index=False)):
            sent_map.setdefault(row.sender_id, []).append(idx)
            recv_map.setdefault(row.receiver_id, []).append(idx)
        self.account_sent_tx_map = sent_map
        self.account_recv_tx_map = recv_map

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class UploadSessionManager:
    """
    Thread-safe registry of active in-memory uploaded datasets mapped to session tokens.
    Automatically purges expired sessions.
    """
    def __init__(self):
        self._datasets: Dict[str, UploadedDataset] = {}
        self._lock = threading.Lock()

    def purge_expired(self) -> int:
        now = datetime.now(timezone.utc)
        with self._lock:
            expired_keys = [k for k, ds in self._datasets.items() if ds.expires_at <= now]
            for k in expired_keys:
                del self._datasets[k]
            return len(expired_keys)

    def store_dataset(self, session_id: str, dataset: UploadedDataset) -> None:
        self.purge_expired()
        with self._lock:
            self._datasets[session_id] = dataset

    def get_dataset(self, session_id: Optional[str]) -> Optional[UploadedDataset]:
        if not session_id:
            return None
        self.purge_expired()
        with self._lock:
            ds = self._datasets.get(session_id)
            if ds and not ds.is_expired():
                return ds
            elif ds and ds.is_expired():
                del self._datasets[session_id]
        return None

    def remove_dataset(self, session_id: str) -> bool:
        with self._lock:
            return self._datasets.pop(session_id, None) is not None

    def active_upload_count(self) -> int:
        now = datetime.now(timezone.utc)
        with self._lock:
            return sum(1 for ds in self._datasets.values() if ds.expires_at > now)


upload_manager = UploadSessionManager()


# ---------------------------------------------------------------------------
# Model Loader Cache
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def get_cached_model() -> Tuple[TGNN, Dict[str, Any]]:
    """Loads and caches the TGNN checkpoint thread-safely."""
    global _MODEL_CACHE
    with _MODEL_LOCK:
        if "model" in _MODEL_CACHE:
            return _MODEL_CACHE["model"], _MODEL_CACHE["ckpt"]

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")

        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = TGNN(
            feat_dim=ckpt["feat_dim"],
            gcn_hidden=ckpt["gcn_hidden"],
            gru_hidden=ckpt["gru_hidden"],
            n_gcn_layers=ckpt["gcn_layers"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["ckpt"] = ckpt
        return model, ckpt


# ---------------------------------------------------------------------------
# CSV Validation & In-Memory Parsing
# ---------------------------------------------------------------------------
def parse_and_validate_csv(raw_bytes: bytes, max_rows: int = MAX_UPLOAD_ROWS) -> pd.DataFrame:
    """
    Validates CSV headers, row bounds, types, positive values, self-transfers,
    and strictly rejects ground-truth fraud columns.
    """
    if len(raw_bytes) == 0:
        raise UploadValidationError("Uploaded CSV file is empty.", "EMPTY_FILE")

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        mb = len(raw_bytes) / (1024 * 1024)
        raise UploadValidationError(
            f"File size ({mb:.2f} MB) exceeds maximum allowed limit of 10 MB.",
            "FILE_TOO_LARGE",
        )

    try:
        csv_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            csv_text = raw_bytes.decode("latin-1")
        except Exception as exc:
            raise UploadValidationError(f"Could not decode file as UTF-8 or Latin-1 text: {str(exc)}", "DECODE_ERROR")

    # Read CSV with pandas in memory
    try:
        df = pd.read_csv(io.StringIO(csv_text), dtype=str)
    except Exception as exc:
        raise UploadValidationError(f"Malformed CSV syntax: {str(exc)}", "MALFORMED_CSV")

    if df.empty or len(df) == 0:
        raise UploadValidationError("CSV contains no data rows.", "NO_ROWS")

    if len(df) > max_rows:
        raise UploadValidationError(
            f"Dataset has {len(df):,} rows, which exceeds the maximum limit of {max_rows:,} rows.",
            "ROW_LIMIT_EXCEEDED",
        )

    # Clean and normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    cols = set(df.columns)

    # 1. Prohibited label check (strict prevention of ground-truth cheating)
    prohibited_found = cols.intersection(PROHIBITED_LABEL_COLUMNS)
    if prohibited_found:
        raise UploadValidationError(
            f"Prohibited label column(s) detected: {sorted(list(prohibited_found))}. "
            "Uploaded data must represent unlabeled operational transactions. Ground-truth fraud columns are strictly rejected.",
            "PROHIBITED_COLUMN",
        )

    # 2. Required columns check
    missing = REQUIRED_COLUMNS - cols
    if missing:
        raise UploadValidationError(
            f"Missing required column(s): {sorted(list(missing))}. "
            "Required columns are: timestamp, sender_id, receiver_id, amount.",
            "MISSING_REQUIRED_COLUMNS",
        )

    # 3. Clean string identifiers
    df["sender_id"] = df["sender_id"].astype(str).str.strip()
    df["receiver_id"] = df["receiver_id"].astype(str).str.strip()

    # Check for empty IDs
    if (df["sender_id"] == "").any() or (df["receiver_id"] == "").any():
        raise UploadValidationError(
            "Encountered empty or blank account IDs in sender_id or receiver_id.",
            "EMPTY_ACCOUNT_ID",
        )

    # 4. Check for self-transfers
    self_tx_mask = df["sender_id"] == df["receiver_id"]
    if self_tx_mask.any():
        bad_count = int(self_tx_mask.sum())
        sample_bad = df.loc[self_tx_mask, "sender_id"].iloc[0]
        raise UploadValidationError(
            f"Detected {bad_count} self-transfer transaction(s) where sender_id equals receiver_id (e.g. '{sample_bad}'). "
            "Self-transfers are invalid in transaction graph topologies.",
            "SELF_TRANSFER_DETECTED",
        )

    # 5. Parse and validate amounts
    try:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    except Exception as exc:
        raise UploadValidationError(f"Could not parse 'amount' column as numerical values: {str(exc)}", "INVALID_AMOUNT")

    if df["amount"].isna().any():
        raise UploadValidationError(
            "Encountered non-numeric or missing values in the 'amount' column.",
            "NON_NUMERIC_AMOUNT",
        )

    # Check positive and finite
    if (df["amount"] <= 0).any() or np.isinf(df["amount"]).any():
        raise UploadValidationError(
            "All transaction amounts must be positive, non-zero, finite numbers (> 0).",
            "NON_POSITIVE_AMOUNT",
        )

    # 6. Parse and validate timestamps
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception as exc:
        raise UploadValidationError(f"Could not parse timestamps: {str(exc)}", "INVALID_TIMESTAMP")

    if df["timestamp"].isna().any():
        raise UploadValidationError(
            "Encountered invalid, unparseable date/time strings in the 'timestamp' column. Use ISO format (YYYY-MM-DD HH:MM:SS).",
            "UNPARSEABLE_TIMESTAMP",
        )

    # Sort strictly chronologically
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Optional columns handling
    if "tx_type" not in df.columns:
        df["tx_type"] = "p2p_transfer"
    else:
        df["tx_type"] = df["tx_type"].astype(str).str.strip().str.lower()
        df["tx_type"] = df["tx_type"].replace("", "p2p_transfer")

    # Generate synthetic transaction_id if not present
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"UP_TX_{i:07d}" for i in range(len(df))]
    else:
        df["transaction_id"] = df["transaction_id"].astype(str).str.strip()

    # Ground-truth fraud is 0 (unlabeled)
    df["is_fraud"] = 0

    return df


# ---------------------------------------------------------------------------
# Account Metadata & Topology Inference
# ---------------------------------------------------------------------------
def extract_accounts_and_types(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Extracts unique accounts and resolves account types (user-provided or inferred).
    """
    all_senders = df["sender_id"].unique()
    all_receivers = df["receiver_id"].unique()
    all_ids = sorted(list(set(all_senders) | set(all_receivers)))

    # If user provided account_type in transaction rows, map the most frequent type per account
    id_to_type: Dict[str, str] = {}
    if "account_type" in df.columns:
        valid_user_types = set(ACCT_TYPES)
        # Check sender types
        for aid, group in df.groupby("sender_id")["account_type"]:
            mode_series = group.mode()
            if not mode_series.empty:
                val = str(mode_series.iloc[0]).strip().lower()
                if val in valid_user_types:
                    id_to_type[aid] = val

    # Infer or fallback for remaining accounts
    # Default topology heuristic: high in-degree accounts act like merchants/agents,
    # high out-degree like disbursers, balanced like beneficiaries
    in_counts = df["receiver_id"].value_counts().to_dict()
    out_counts = df["sender_id"].value_counts().to_dict()

    acct_rows = []
    for aid in all_ids:
        atype = id_to_type.get(aid)
        if not atype:
            inc = in_counts.get(aid, 0)
            outc = out_counts.get(aid, 0)
            if inc > 50 and outc < 10:
                atype = "merchant"
            elif inc > 20 and outc > 20:
                atype = "agent"
            else:
                atype = "beneficiary"
            id_to_type[aid] = atype

        acct_rows.append({
            "account_id": aid,
            "account_type": atype,
            "is_fraud_account": 0,  # Unlabeled
        })

    acct_df = pd.DataFrame(acct_rows)
    return acct_df, id_to_type


# ---------------------------------------------------------------------------
# TGNN Feature Engineering & Inference for Uploaded Data
# ---------------------------------------------------------------------------
def score_uploaded_data(
    txn_df: pd.DataFrame,
    acct_df: pd.DataFrame,
    id_to_type: Dict[str, str],
) -> pd.DataFrame:
    """
    Executes TGNN graph snapshot construction and forward pass inference on uploaded data,
    reusing the saved model checkpoint and scaler parameters.
    Handles short temporal sequences by repeating the available snapshots to meet the 6-step GRU requirement.
    """
    model, ckpt = get_cached_model()

    all_ids = sorted(acct_df["account_id"].unique())
    id2idx = {aid: i for i, aid in enumerate(all_ids)}
    N = len(all_ids)

    # Map sender/receiver indices
    sender_idx = txn_df["sender_id"].map(id2idx).values.astype(np.int32)
    receiver_idx = txn_df["receiver_id"].map(id2idx).values.astype(np.int32)
    amounts = txn_df["amount"].values.astype(np.float32)

    # 1. Build Static Account Type Tensor [N, 4]
    type2col = {t: i for i, t in enumerate(ACCT_TYPES)}
    acct_mat = np.zeros((N, len(ACCT_TYPES)), dtype=np.float32)
    for i, aid in enumerate(all_ids):
        t = id_to_type.get(aid, "beneficiary")
        col = type2col.get(t, 1)  # fallback to beneficiary (col 1)
        acct_mat[i, col] = 1.0
    acct_type_tensor = torch.tensor(acct_mat, dtype=torch.float32)

    # 2. Build 30-minute Graph Snapshots
    t_min = txn_df["timestamp"].min()
    window_td = pd.Timedelta(minutes=WINDOW_MINUTES)
    window_indices = ((txn_df["timestamp"] - t_min) // window_td).astype(int).values
    n_windows = max(1, int(window_indices.max()) + 1)

    # Scaler mean & scale from checkpoint
    scaler_mean = ckpt["scaler_mean"]
    scaler_scale = ckpt["scaler_scale"]

    raw_behav_list = []
    adj_list = []

    for w in range(n_windows):
        mask = (window_indices == w)
        s = sender_idx[mask]
        d = receiver_idx[mask]
        a = amounts[mask]

        behav = build_snapshot_features(s, d, a, N)
        # Apply scaler safely with numerical bounding
        scaled_behav = (behav - scaler_mean) / (scaler_scale + 1e-9)
        # Numerical safeguard against extreme out-of-distribution bursts
        scaled_behav = np.clip(scaled_behav, -15.0, 15.0)

        raw_behav_list.append(behav)

        adj = build_sparse_adj(s, d, a, N)
        adj_list.append(adj)

    # Combine scaled behavior + account type
    snapshots = []
    for w in range(n_windows):
        behav_t = torch.tensor(
            (raw_behav_list[w] - scaler_mean) / (scaler_scale + 1e-9),
            dtype=torch.float32,
        )
        behav_t = torch.clamp(behav_t, -15.0, 15.0)
        feats = torch.cat([behav_t, acct_type_tensor], dim=1)
        snapshots.append((feats, adj_list[w]))

    # 3. Model Inference over Snapshots
    max_scores = np.zeros(N, dtype=np.float64)
    peak_features = np.zeros((N, BEHAV_FEAT_DIM), dtype=np.float32)

    with torch.no_grad():
        for t_abs in range(n_windows):
            seq_indices = list(range(max(0, t_abs - N_SNAPSHOTS_SEQ + 1), t_abs + 1))
            while len(seq_indices) < N_SNAPSHOTS_SEQ:
                seq_indices.insert(0, seq_indices[0])

            seq_f = [snapshots[j][0] for j in seq_indices]
            seq_a = [snapshots[j][1] for j in seq_indices]

            probs = model(seq_f, seq_a).cpu().numpy()
            updated = probs > max_scores
            max_scores[updated] = probs[updated]

            cur_behav = raw_behav_list[t_abs]
            peak_features[updated] = cur_behav[updated]

    # 4. Extract Top Driving Signals & Rule Flags
    # Static rule baseline: p99 amount or velocity burst > 20 tx / 30 min
    p99_val = float(txn_df["amount"].quantile(0.99))
    large_senders = set(txn_df.loc[txn_df["amount"] > p99_val, "sender_id"].unique())

    # Burst senders in any 30-min window
    burst_senders: Set[str] = set()
    for w in range(n_windows):
        mask = (window_indices == w)
        if mask.any():
            w_df = txn_df[mask]
            counts = w_df["sender_id"].value_counts()
            burst_senders.update(counts[counts > 20].index.tolist())

    rule_flagged_set = large_senders | burst_senders

    feature_names = [
        "out_count",
        "in_count",
        "log_out_amount",
        "log_in_amount",
        "forwarding_ratio",
        "unique_receivers",
        "unique_senders",
        "amount_std",
    ]

    scores_rows = []
    for i, aid in enumerate(all_ids):
        t_score = float(max_scores[i])
        r_flag = 1 if aid in rule_flagged_set else 0

        # Determine top 3 driving features by z-score deviation
        row_z = (peak_features[i] - scaler_mean) / (scaler_scale + 1e-9)
        top_indices = np.argsort(row_z)[::-1][:3]
        top_sigs = [feature_names[idx] for idx in top_indices if row_z[idx] > 0.0]
        while len(top_sigs) < 3:
            top_sigs.append("none")

        scores_rows.append({
            "account_id": aid,
            "tgnn_score": t_score,
            "rule_flagged": r_flag,
            "lr_score": None,  # Not applicable on arbitrary uploads
            "hybrid_score": None,
            "true_label": 0,   # No ground truth on uploaded data
            "top_signal_1": top_sigs[0],
            "top_signal_2": top_sigs[1],
            "top_signal_3": top_sigs[2],
        })

    scores_df = pd.DataFrame(scores_rows)
    return scores_df


# ---------------------------------------------------------------------------
# Upload Pipeline Orchestrator
# ---------------------------------------------------------------------------
def process_uploaded_csv(
    session_id: str,
    raw_bytes: bytes,
    filename: str = "uploaded.csv",
) -> UploadedDataset:
    """
    End-to-end pipeline: validates CSV, extracts graph accounts, scores via TGNN,
    computes dataset summaries, and stores in the per-session isolated manager.
    """
    # 1. Validate & Parse CSV
    txn_df = parse_and_validate_csv(raw_bytes)

    # 2. Extract accounts & topology types
    acct_df, id_to_type = extract_accounts_and_types(txn_df)

    # 3. Model scoring
    scores_df = score_uploaded_data(txn_df, acct_df, id_to_type)

    # 4. Precompute Dataset Summary
    total_tx = len(txn_df)
    total_vol = float(txn_df["amount"].sum())
    total_accts = len(acct_df)

    ts_min = txn_df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S")
    ts_max = txn_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")

    # Format timestamps as strings matching standard transaction schema
    txn_df["timestamp"] = txn_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "dataset_name": filename,
        "is_uploaded": True,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "total_accounts": total_accts,
        "total_transactions": total_tx,
        "total_volume": round(total_vol, 2),
        "average_transaction_amount": round(total_vol / total_tx, 2) if total_tx > 0 else 0.0,
        "fraud_transactions_count": 0,  # Explicitly unlabeled
        "fraud_transaction_pct": 0.0,
        "fraud_volume": 0.0,
        "fraud_volume_pct": 0.0,
        "simulation_window": {
            "start": ts_min,
            "end": ts_max,
        },
        "account_types": acct_df["account_type"].value_counts().to_dict(),
        "ground_truth_fraud_accounts": 0,
        "notice": (
            "Custom uploaded dataset analyzed in-memory. Risk scores represent exploratory "
            "topological graph estimates from the TGNN model trained on synthetic G2P data. "
            "No ground-truth fraud labels exist for uploaded data."
        ),
    }

    dataset = UploadedDataset(
        session_id=session_id,
        transactions_df=txn_df,
        accounts_df=acct_df,
        scores_df=scores_df,
        summary=summary,
        ttl_minutes=UPLOAD_TTL_MINUTES,
    )

    upload_manager.store_dataset(session_id, dataset)
    return dataset


# ---------------------------------------------------------------------------
# Sample CSV Generator
# ---------------------------------------------------------------------------
def generate_sample_csv() -> str:
    """
    Generates a valid, compliant sample CSV with required and optional columns
    demonstrating rapid pass-through and fan-out topologies without fraud labels.
    """
    output = io.StringIO()
    output.write("timestamp,sender_id,receiver_id,amount,account_type,tx_type\n")
    
    # 25 realistic sample transactions
    base_time = datetime(2026, 10, 16, 10, 0, 0)
    samples = [
        (0, "ACC_GOV_01", "ACC_BENEF_101", 5000.00, "government", "welfare_payout"),
        (2, "ACC_GOV_01", "ACC_BENEF_102", 5000.00, "government", "welfare_payout"),
        (5, "ACC_GOV_01", "ACC_BENEF_103", 5000.00, "government", "welfare_payout"),
        (8, "ACC_GOV_01", "ACC_BENEF_104", 5000.00, "government", "welfare_payout"),
        (15, "ACC_BENEF_101", "ACC_MERCH_501", 125.50, "beneficiary", "retail_purchase"),
        (18, "ACC_BENEF_102", "ACC_MERCH_501", 89.20, "beneficiary", "retail_purchase"),
        # Rapid pass-through / mule behavior
        (25, "ACC_BENEF_103", "ACC_MULE_901", 4800.00, "beneficiary", "p2p_transfer"),
        (28, "ACC_BENEF_104", "ACC_MULE_901", 4900.00, "beneficiary", "p2p_transfer"),
        (35, "ACC_MULE_901", "ACC_AGENT_701", 4700.00, "beneficiary", "agent_cash_out"),
        (38, "ACC_MULE_901", "ACC_AGENT_702", 4800.00, "beneficiary", "agent_cash_out"),
        # Normal utility and retail
        (42, "ACC_BENEF_101", "ACC_UTIL_601", 450.00, "beneficiary", "utility_payment"),
        (45, "ACC_BENEF_102", "ACC_UTIL_601", 320.00, "beneficiary", "utility_payment"),
        (50, "ACC_BENEF_105", "ACC_MERCH_502", 1500.00, "beneficiary", "retail_purchase"),
        (55, "ACC_BENEF_106", "ACC_MERCH_502", 210.00, "beneficiary", "retail_purchase"),
        (60, "ACC_GOV_01", "ACC_BENEF_105", 5000.00, "government", "welfare_payout"),
        (62, "ACC_GOV_01", "ACC_BENEF_106", 5000.00, "government", "welfare_payout"),
        (70, "ACC_BENEF_105", "ACC_AGENT_701", 3000.00, "beneficiary", "agent_cash_out"),
        (75, "ACC_BENEF_106", "ACC_AGENT_702", 4000.00, "beneficiary", "agent_cash_out"),
        (85, "ACC_AGENT_701", "ACC_MERCH_501", 1200.00, "agent", "p2p_transfer"),
        (90, "ACC_AGENT_702", "ACC_MERCH_502", 1800.00, "agent", "p2p_transfer"),
    ]

    for delta_m, sender, receiver, amt, atype, txtype in samples:
        t_str = (base_time + timedelta(minutes=delta_m)).strftime("%Y-%m-%d %H:%M:%S")
        output.write(f"{t_str},{sender},{receiver},{amt:.2f},{atype},{txtype}\n")

    return output.getvalue()
