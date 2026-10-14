"""
==============================================================================
PulseGuard-Z  -  Phase 2: Evaluation & Scoring  (v2 -- Part A dataset)
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : ml/evaluate.py

Purpose
-------
Evaluates the trained TGNN against a rule-based baseline and a logistic
regression baseline on the held-out test split of the Part-A (v2) dataset.

Pipeline
--------
1. Load data/transactions.csv, data/accounts.csv, data/ring_membership.csv,
   and ml/tgnn_model.pt.
2. Reconstruct graph snapshots using the same builder as training (including
   account-type one-hot features).
3. Derive ground-truth labels for validation and test periods.
4. Run TGNN inference across validation snapshots -> select operating threshold
   tau_TGNN by maximising F1 on validation.
5. Rule-based baseline:
     - Amount > 99th-percentile of TRAINING transactions (no future leakage).
     - Rapid transfer velocity (> 20 transactions in a 30-min window).
   No threshold to tune (binary output).
6. Logistic Regression baseline:
     - Features: mean + max of 8 behavioural node features across train
       snapshots, plus 4 account-type one-hot dims = 20 features total.
     - Fit with class_weight='balanced'; tune regularisation C on validation.
     - Select threshold tau_LR by maximising F1 on validation.
7. Evaluate all three models on the held-out TEST set ONCE using the
   thresholds chosen on validation.
8. Report: Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrices,
   FP breakdown by account_type, ring detection rate.
9. Save data/scores.csv, docs/roc_curve.png, docs/pr_curve.png,
   docs/results.md.

Threshold-selection protocol
-----------------------------
- Thresholds for TGNN and LR are chosen on VALIDATION data only.
- The test set is evaluated EXACTLY ONCE using those thresholds.
- Rule baseline is binary; threshold = 0.5 by definition.

Usage
-----
    python ml/evaluate.py
==============================================================================
"""

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

# Silence benign PyTorch sparse invariant warning on CPU
torch.sparse.check_sparse_tensor_invariants.disable()

# ---------------------------------------------------------------------------
# Paths & Imports from train_tgnn
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
ML_DIR     = REPO_ROOT / "ml"
DOCS_DIR   = REPO_ROOT / "docs"

TXN_CSV          = DATA_DIR / "transactions.csv"
ACCT_CSV         = DATA_DIR / "accounts.csv"
RING_CSV         = DATA_DIR / "ring_membership.csv"
MODEL_PATH       = ML_DIR   / "tgnn_model.pt"
SCORES_CSV       = DATA_DIR / "scores.csv"
ROC_PNG          = DOCS_DIR / "roc_curve.png"
PR_PNG           = DOCS_DIR / "pr_curve.png"
RESULTS_MD       = DOCS_DIR / "results.md"

sys.path.insert(0, str(REPO_ROOT))
from ml.train_tgnn import (
    NODE_FEAT_DIM,
    BEHAV_FEAT_DIM,
    ACCT_TYPES,
    WINDOW_MINUTES,
    N_SNAPSHOTS_SEQ,
    TRAIN_FRAC,
    VAL_FRAC,
    build_snapshot_features,
    build_sparse_adj,
    build_all_snapshots,
    build_account_type_matrix,
    TGNN,
)

# Behavioural feature names (first BEHAV_FEAT_DIM columns of node feature vector)
BEHAV_FEATURE_NAMES = [
    "out_count", "in_count", "log_out_amount", "log_in_amount",
    "forwarding_ratio", "unique_receivers", "unique_senders", "amount_std",
]


# ---------------------------------------------------------------------------
# 1.  Load Data
# ---------------------------------------------------------------------------

def load_data():
    print("[Data] Loading CSVs ...")
    txn  = pd.read_csv(TXN_CSV,  parse_dates=["timestamp"])
    acct = pd.read_csv(ACCT_CSV)
    txn.sort_values("timestamp", inplace=True)
    txn.reset_index(drop=True, inplace=True)

    all_ids = sorted(acct["account_id"].unique())
    id2idx  = {aid: i for i, aid in enumerate(all_ids)}
    N       = len(all_ids)

    txn["sender_idx"]   = txn["sender_id"].map(id2idx).astype(np.int32)
    txn["receiver_idx"] = txn["receiver_id"].map(id2idx).astype(np.int32)

    acct_type_tensor = build_account_type_matrix(acct, all_ids)
    return txn, acct, id2idx, N, all_ids, acct_type_tensor


def load_ring_membership():
    if not RING_CSV.exists():
        print("[Ring] ring_membership.csv not found -- ring detection skipped.")
        return None
    rm = pd.read_csv(RING_CSV)
    print(f"[Ring] Loaded ring_membership.csv: "
          f"{len(rm)} rows, {rm['ring_id'].nunique()} rings")
    return rm


# ---------------------------------------------------------------------------
# 2.  Load Model
# ---------------------------------------------------------------------------

def load_model():
    ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = TGNN(
        feat_dim=ckpt["feat_dim"],
        gcn_hidden=ckpt["gcn_hidden"],
        gru_hidden=ckpt["gru_hidden"],
        n_gcn_layers=ckpt["gcn_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Model] Loaded  best_val_loss={ckpt['best_val_loss']:.4f}  "
          f"pos_weight={ckpt['pos_weight']:.1f}  feat_dim={ckpt['feat_dim']}")
    return model, ckpt


# ---------------------------------------------------------------------------
# 3.  Ground-Truth Labels
# ---------------------------------------------------------------------------

def period_labels(txn, all_ids, t_start, t_end):
    """
    Per-account fraud label for a monitoring period [t_start, t_end).
    An account is labeled 1 if it is sender OR receiver of any fraud
    transaction whose timestamp falls in [t_start, t_end).
    """
    period = txn[(txn["timestamp"] >= t_start) & (txn["timestamp"] < t_end)]
    fraud_ids = (
        set(period.loc[period["is_fraud"] == 1, "sender_id"])
        | set(period.loc[period["is_fraud"] == 1, "receiver_id"])
    )
    y = np.array([1.0 if aid in fraud_ids else 0.0 for aid in all_ids])
    return y


# ---------------------------------------------------------------------------
# 4.  TGNN Inference (generic: any contiguous window range)
# ---------------------------------------------------------------------------

def tgnn_inference(model, all_snapshots, start_idx, end_idx, N):
    """
    Run TGNN inference from snapshot start_idx (inclusive) to end_idx
    (exclusive), with a sliding GRU look-back of N_SNAPSHOTS_SEQ.

    Returns:
      max_scores    : [N] per-account peak fraud probability over the window
      peak_features : [N, BEHAV_FEAT_DIM] behavioural features at peak snapshot
    """
    model.eval()
    max_scores    = np.zeros(N, dtype=np.float64)
    peak_features = np.zeros((N, BEHAV_FEAT_DIM), dtype=np.float32)

    with torch.no_grad():
        for t_abs in range(start_idx, end_idx):
            seq_indices = list(range(max(0, t_abs - N_SNAPSHOTS_SEQ + 1), t_abs + 1))
            while len(seq_indices) < N_SNAPSHOTS_SEQ:
                seq_indices.insert(0, seq_indices[0])

            seq_f = [all_snapshots[j][0] for j in seq_indices]
            seq_a = [all_snapshots[j][1] for j in seq_indices]
            probs = model(seq_f, seq_a).numpy()

            updated = probs > max_scores
            max_scores[updated] = probs[updated]
            # Store only the BEHAVIOURAL part of the feature vector for scores.csv
            cur_behav = all_snapshots[t_abs][0].numpy()[:, :BEHAV_FEAT_DIM]
            peak_features[updated] = cur_behav[updated]

    return max_scores, peak_features


# ---------------------------------------------------------------------------
# 5.  Rule-Based Baseline
# ---------------------------------------------------------------------------

def rule_baseline(txn, all_ids, N, p99_train, t_period_start, t_period_end):
    """
    Rule-based binary flags for a given monitoring period.

    p99_train: 99th-percentile amount threshold computed from TRAINING
               transactions only (no look-ahead into val/test data).

    Rules:
      1. Any account that sent a transaction with amount > p99_train.
      2. Any account that sent > 20 transactions in a single 30-min window.
    """
    period = txn[(txn["timestamp"] >= t_period_start) & (txn["timestamp"] < t_period_end)]
    flagged = set(period.loc[period["amount"] > p99_train, "sender_id"].unique())

    if len(period) > 0:
        t_min_p   = period["timestamp"].min()
        t_max_p   = period["timestamp"].max()
        window_td = pd.Timedelta(minutes=WINDOW_MINUTES)
        cur = t_min_p
        while cur <= t_max_p:
            mask   = (period["timestamp"] >= cur) & (period["timestamp"] < cur + window_td)
            counts = period[mask].groupby("sender_id").size()
            flagged.update(counts[counts > 20].index.tolist())
            cur += window_td

    scores = np.array([1.0 if aid in flagged else 0.0 for aid in all_ids])
    return scores


# ---------------------------------------------------------------------------
# 6.  Logistic Regression Baseline
# ---------------------------------------------------------------------------

def aggregate_snap_features(snap_list, N):
    """
    Aggregate behavioural node features across a list of snapshots.
    Returns mean and max of the BEHAV_FEAT_DIM=8 behavioural features per account.
    Shape: [N, BEHAV_FEAT_DIM * 2] = [N, 16].
    """
    feat_sum = np.zeros((N, BEHAV_FEAT_DIM), dtype=np.float64)
    feat_max = np.zeros((N, BEHAV_FEAT_DIM), dtype=np.float64)
    n_snaps  = len(snap_list)
    for feats, adj, labels in snap_list:
        f = feats.numpy()[:, :BEHAV_FEAT_DIM]   # only behavioural dims
        feat_sum += f
        feat_max  = np.maximum(feat_max, f)
    feat_mean = feat_sum / max(n_snaps, 1)
    return np.concatenate([feat_mean, feat_max], axis=1).astype(np.float32)


def build_lr_features(snap_list, N, acct_type_arr):
    """
    Full LR feature matrix: aggregate behavioural (16) + account-type one-hot (4).
    Shape: [N, 20].  Static account_type is appended last.
    """
    behav_agg = aggregate_snap_features(snap_list, N)
    return np.concatenate([behav_agg, acct_type_arr], axis=1).astype(np.float32)


def snap_period_labels(snap_list, N):
    """Per-account label: 1 if flagged in any snapshot in snap_list."""
    labels = torch.zeros(N, dtype=torch.float32)
    for _, _, lbl in snap_list:
        labels = torch.clamp(labels + lbl, max=1.0)
    return labels.numpy()


def train_lr(X_train, y_train, X_val, y_val):
    """
    Train LogisticRegression with class_weight='balanced'.
    Tune regularisation C in {0.01, 0.1, 1.0, 10.0} using val F1.
    Returns (model, scaler, best_C).
    """
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    best_f1 = -1.0
    best_C  = 1.0
    best_lr = None

    for C in [0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(
            C=C, class_weight="balanced",
            max_iter=1000, random_state=42, solver="lbfgs",
        )
        lr.fit(X_tr_sc, y_train)
        preds = lr.predict(X_va_sc)
        f1    = f1_score(y_val, preds, zero_division=0)
        print(f"  [LR] C={C:.2f}  val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_C  = C
            best_lr = lr

    print(f"  [LR] Best C={best_C}  val_F1={best_f1:.4f}")
    return best_lr, scaler, best_C


# ---------------------------------------------------------------------------
# 7.  Threshold Selection on Validation
# ---------------------------------------------------------------------------

def select_threshold(y_val, val_scores, model_name="Model"):
    """
    Sweep threshold tau in [0.01, 0.99] step 0.01.
    Choose tau* = argmax F1 on y_val.
    Returns (tau*, best_val_F1).
    """
    best_tau = 0.50
    best_f1  = -1.0
    for tau in np.arange(0.01, 1.00, 0.01):
        preds = (val_scores >= tau).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)
    print(f"  [{model_name}] Val threshold: tau*={best_tau:.2f}  val_F1={best_f1:.4f}")
    return best_tau, best_f1


# ---------------------------------------------------------------------------
# 8.  Compute Metrics
# ---------------------------------------------------------------------------

def compute_metrics(name, y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    acc    = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n  [{name}]  threshold={threshold:.2f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-score  : {f1:.4f}")
    print(f"    ROC-AUC   : {auc:.4f}")
    print(f"    PR-AUC    : {pr_auc:.4f}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Confusion matrix (rows=actual, cols=pred):")
    print(f"      TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "name": name, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": auc, "pr_auc": pr_auc, "accuracy": acc, "cm": cm,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# 9.  False Positive Analysis by account_type
# ---------------------------------------------------------------------------

def fp_by_account_type(y_true, y_pred, all_ids, acct_df):
    """Return a dict {account_type: fp_count} for the given binary prediction."""
    id2type = dict(zip(acct_df["account_id"], acct_df["account_type"]))
    fp_counts = {}
    for i, aid in enumerate(all_ids):
        if y_pred[i] == 1 and y_true[i] == 0:            # false positive
            atype = id2type.get(aid, "unknown")
            fp_counts[atype] = fp_counts.get(atype, 0) + 1
    return fp_counts


# ---------------------------------------------------------------------------
# 10.  Ring Detection Rate
# ---------------------------------------------------------------------------

def ring_detection_rate(y_pred_binary, all_ids, ring_membership, txn, t_test_start):
    """
    Detection definition:
      - A ring is 'active in test' if ANY of its member accounts (mules) appears
        as sender OR receiver in at least one fraud transaction with
        timestamp >= t_test_start.
      - A ring is 'detected' if at least one of its member accounts is flagged
        (y_pred == 1) by the model.

    Returns (rate, n_detected, n_active).
    """
    if ring_membership is None:
        return float("nan"), 0, 0

    id2pred = {aid: int(y_pred_binary[i]) for i, aid in enumerate(all_ids)}

    test_fraud = txn[(txn["timestamp"] >= t_test_start) & (txn["is_fraud"] == 1)]
    test_participants = (
        set(test_fraud["sender_id"]) | set(test_fraud["receiver_id"])
    )

    active_rings   = set()
    detected_rings = set()

    for ring_id, group in ring_membership.groupby("ring_id"):
        mule_ids = set(group["account_id"])
        if mule_ids & test_participants:          # ring has activity in test
            active_rings.add(ring_id)
            if any(id2pred.get(m, 0) == 1 for m in mule_ids):
                detected_rings.add(ring_id)

    n_active   = len(active_rings)
    n_detected = len(detected_rings)
    rate       = n_detected / max(n_active, 1)
    print(f"  Ring detection: {n_detected}/{n_active} rings detected "
          f"(rate={rate:.3f})")
    return rate, n_detected, n_active


# ---------------------------------------------------------------------------
# 11.  Visualisations
# ---------------------------------------------------------------------------

def save_roc_chart(models_data, y_true):
    """models_data: list of (name, scores, color, linestyle)"""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.set_facecolor("white")

    for name, scores, color, ls in models_data:
        try:
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_val     = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=color, lw=2.0, linestyle=ls,
                    label=f"{name} (AUC={auc_val:.4f})")
        except Exception as e:
            print(f"[ROC] Could not plot {name}: {e}")

    ax.plot([0, 1], [0, 1], color="#8d8d8d", lw=1.0, linestyle=":", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("PulseGuard-Z: ROC Curve (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROC_PNG, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {ROC_PNG}")


def save_pr_chart(models_data, y_true):
    """models_data: list of (name, scores, color, linestyle)"""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.set_facecolor("white")

    for name, scores, color, ls in models_data:
        try:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(rec_arr, prec_arr, color=color, lw=2.0, linestyle=ls,
                    label=f"{name} (AP={ap:.4f})")
        except Exception as e:
            print(f"[PR] Could not plot {name}: {e}")

    baseline_prec = float(np.sum(y_true)) / len(y_true)
    ax.axhline(baseline_prec, color="#8d8d8d", lw=1.0, linestyle=":",
               label=f"Random ({baseline_prec:.4f})")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("PulseGuard-Z: Precision-Recall Curve (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PR_PNG, dpi=150)
    plt.close(fig)
    print(f"[Chart] Saved: {PR_PNG}")


# ---------------------------------------------------------------------------
# 12.  Top Signals & CSV Export
# ---------------------------------------------------------------------------

def top3_signals(feat_row):
    order = np.argsort(np.abs(feat_row))[::-1]
    return [BEHAV_FEATURE_NAMES[i] for i in order[:3]]


def save_scores_csv(all_ids, tgnn_scores, rule_scores, lr_scores, y_true, peak_features):
    """
    Save data/scores.csv.  Columns required by backend:
      account_id, tgnn_score, rule_flagged, true_label,
      top_signal_1, top_signal_2, top_signal_3.
    Additional columns: lr_score.
    """
    rows = []
    for i, aid in enumerate(all_ids):
        sig1, sig2, sig3 = top3_signals(peak_features[i])
        rows.append({
            "account_id":   aid,
            "tgnn_score":   float(tgnn_scores[i]),
            "rule_flagged": int(rule_scores[i]),
            "lr_score":     float(lr_scores[i]),
            "true_label":   int(y_true[i]),
            "top_signal_1": sig1,
            "top_signal_2": sig2,
            "top_signal_3": sig3,
        })
    df = pd.DataFrame(rows)
    df.sort_values("tgnn_score", ascending=False, inplace=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)
    print(f"[Scores] Saved: {SCORES_CSV} ({len(df):,} accounts)")
    return df


# ---------------------------------------------------------------------------
# 13.  Write results.md
# ---------------------------------------------------------------------------

TEMPLATE = """\
# PulseGuard-Z — Phase 2 Evaluation Results (v2 Dataset)
**IBM Z Hackathon | SSF2026 | Team: Error-404**

---

## Data Summary

| Metric | Value |
|--------|-------|
| Total accounts | {n_accts:,} |
| Total transactions | {n_txns:,} |
| Fraud transactions | {n_fraud:,} ({fraud_pct:.2f}%) |
| Simulation period | {t_min} → {t_max} |
| Train-period fraud accounts | {n_fraud_accts_train} |
| Val-period fraud accounts | {n_fraud_accts_val} |
| Test-period fraud accounts | {n_fraud_accts_test} |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Snapshot window | 30 minutes |
| GRU look-back | 6 snapshots (3 hours) |
| Node features total | 12 (8 behavioural + 4 account-type one-hot) |
| GCN layers × hidden | 2 × 32 |
| GRU hidden | 64 |
| Dropout | 0.30 |
| Train / Val / Test split | 70 / 15 / 15 % (chronological) |
| Weighted BCE pos_weight | ~{pos_weight:.1f} |
| Best validation loss | {best_val_loss:.4f} |
| LR regularisation C | {lr_C} |

---

## Threshold Selection (Validation Set)

Thresholds chosen by maximising F1 on validation data. Test set evaluated once.

| Model | Val threshold (τ*) | Val F1 |
|-------|--------------------|--------|
| TGNN | {tgnn_tau:.2f} | {tgnn_val_f1:.4f} |
| Logistic Regression | {lr_tau:.2f} | {lr_val_f1:.4f} |
| Rule Baseline | 0.50 (binary) | — |

---

## Test-Set Metrics

> Positive test examples: **{n_pos_test}** / {n_accts} ({pos_pct:.2f}%)

| Model | τ | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy |
|-------|---|-----------|--------|----|---------|--------|----------|
| **TGNN** | {tgnn_tau:.2f} | {tgnn_prec:.4f} | {tgnn_rec:.4f} | {tgnn_f1:.4f} | **{tgnn_auc:.4f}** | {tgnn_pr_auc:.4f} | {tgnn_acc:.4f} |
| **Logistic Regression** | {lr_tau:.2f} | {lr_prec:.4f} | {lr_rec:.4f} | {lr_f1:.4f} | {lr_auc:.4f} | {lr_pr_auc:.4f} | {lr_acc:.4f} |
| Rule Baseline | 0.50 | {rule_prec:.4f} | {rule_rec:.4f} | {rule_f1:.4f} | {rule_auc:.4f} | {rule_pr_auc:.4f} | {rule_acc:.4f} |

---

## Confusion Matrices

### TGNN (τ={tgnn_tau:.2f})
```
             Predicted 0  Predicted 1
Actual 0     {tgnn_tn:10d} {tgnn_fp:12d}
Actual 1     {tgnn_fn:10d} {tgnn_tp:12d}
```

### Logistic Regression (τ={lr_tau:.2f})
```
             Predicted 0  Predicted 1
Actual 0     {lr_tn:10d} {lr_fp:12d}
Actual 1     {lr_fn:10d} {lr_tp:12d}
```

### Rule Baseline (τ=0.50)
```
             Predicted 0  Predicted 1
Actual 0     {rule_tn:10d} {rule_fp:12d}
Actual 1     {rule_fn:10d} {rule_tp:12d}
```

---

## False Positives by Account Type

{fp_table}

---

## Fraud Ring Detection Rate

> **Detection definition:** A ring is *active in test* if any of its mule accounts
> appears in a fraud transaction during the test period. A ring is *detected* if
> at least one of its mule accounts is flagged (predicted positive) by the model.

| Model | Rings Active (test) | Rings Detected | Detection Rate |
|-------|---------------------|----------------|----------------|
| TGNN | {ring_active} | {tgnn_ring_det} | {tgnn_ring_rate:.3f} |
| Logistic Regression | {ring_active} | {lr_ring_det} | {lr_ring_rate:.3f} |
| Rule Baseline | {ring_active} | {rule_ring_det} | {rule_ring_rate:.3f} |

---

## ROC Curve
![ROC Curve](roc_curve.png)

## Precision-Recall Curve
![PR Curve](pr_curve.png)

---

## Top Signals (Highest TGNN Risk Scores)

| Account | TGNN Score | True Label | Signal 1 | Signal 2 | Signal 3 |
|---------|------------|------------|----------|----------|----------|
{top_rows}

---

## Limitations & Honest Assessment

> [!WARNING]
> The following limitations must be understood before deploying this model.

1. **Synthetic data.** Fraud patterns were planted deliberately. Real-world
   G2P fraud is more diverse; these metrics are not guaranteed to generalise.

2. **Test class imbalance.** The test period contains {n_pos_test} fraud-active
   accounts out of {n_accts}. Precision is sensitive to false-positive count.

3. **Probability calibration.** TGNN sigmoid outputs are uncalibrated due to
   the heavy positive loss weighting. Decision thresholds should be tuned to
   the operational alert queue budget.

4. **Receiver labeling.** Both sender AND receiver of a fraud transaction are
   labeled positive in training. Innocent recipients of ring hops contribute
   some noise to the positive class.

5. **Temporal granularity.** 30-minute snapshot windows aggregate sub-minute
   intra-burst dynamics into summary statistics.

6. **Model scale.** GCN hidden=32, GRU hidden=64 — intentionally lightweight
   for CPU execution and mainframe deployment.

7. **Transparent comparison.** All three models are evaluated on the identical
   held-out test partition using thresholds selected on validation data only.

---

*Generated by `ml/evaluate.py`  |  PulseGuard-Z Phase 2 (Part A v2 dataset)*
"""


def _cm4(res):
    cm = res["cm"]
    return cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]


def _fp_table_str(tgnn_fp_by_type, lr_fp_by_type, rule_fp_by_type, acct_types_all):
    lines = ["| Account Type | TGNN FP | LR FP | Rule FP |",
             "|---|---|---|---|"]
    for t in sorted(acct_types_all):
        lines.append(
            f"| {t} | {tgnn_fp_by_type.get(t, 0)} "
            f"| {lr_fp_by_type.get(t, 0)} "
            f"| {rule_fp_by_type.get(t, 0)} |"
        )
    return "\n".join(lines)


def write_results_md(
    txn, acct, y_val, y_true,
    tgnn_res, lr_res, rule_res,
    tgnn_tau, lr_tau, tgnn_val_f1, lr_val_f1,
    ckpt, lr_C,
    tgnn_fp_by_type, lr_fp_by_type, rule_fp_by_type,
    tgnn_ring, lr_ring, rule_ring,
    ring_active,
    n_fraud_accts_train, n_fraud_accts_val,
    top_rows_str,
):
    n_accts   = len(acct)
    n_txns    = len(txn)
    n_fraud   = int((txn["is_fraud"] == 1).sum())
    fraud_pct = n_fraud / n_txns * 100
    t_min     = str(txn["timestamp"].min())[:19]
    t_max     = str(txn["timestamp"].max())[:19]
    n_pos     = int(y_true.sum())
    pos_pct   = n_pos / n_accts * 100

    acct_types_all = set(acct["account_type"].unique())
    fp_table = _fp_table_str(
        tgnn_fp_by_type, lr_fp_by_type, rule_fp_by_type, acct_types_all
    )

    tn1, fp1, fn1, tp1 = _cm4(tgnn_res)
    tn2, fp2, fn2, tp2 = _cm4(lr_res)
    tn3, fp3, fn3, tp3 = _cm4(rule_res)

    md = TEMPLATE.format(
        n_accts=n_accts, n_txns=n_txns, n_fraud=n_fraud, fraud_pct=fraud_pct,
        t_min=t_min, t_max=t_max,
        n_fraud_accts_train=n_fraud_accts_train,
        n_fraud_accts_val=n_fraud_accts_val,
        n_fraud_accts_test=n_pos,
        pos_weight=ckpt["pos_weight"], best_val_loss=ckpt["best_val_loss"],
        lr_C=lr_C,
        tgnn_tau=tgnn_tau, tgnn_val_f1=tgnn_val_f1,
        lr_tau=lr_tau,     lr_val_f1=lr_val_f1,
        n_pos_test=n_pos, pos_pct=pos_pct,
        tgnn_prec=tgnn_res["precision"], tgnn_rec=tgnn_res["recall"],
        tgnn_f1=tgnn_res["f1"], tgnn_auc=tgnn_res["roc_auc"],
        tgnn_pr_auc=tgnn_res["pr_auc"], tgnn_acc=tgnn_res["accuracy"],
        lr_prec=lr_res["precision"], lr_rec=lr_res["recall"],
        lr_f1=lr_res["f1"], lr_auc=lr_res["roc_auc"],
        lr_pr_auc=lr_res["pr_auc"], lr_acc=lr_res["accuracy"],
        rule_prec=rule_res["precision"], rule_rec=rule_res["recall"],
        rule_f1=rule_res["f1"], rule_auc=rule_res["roc_auc"],
        rule_pr_auc=rule_res["pr_auc"], rule_acc=rule_res["accuracy"],
        tgnn_tn=tn1, tgnn_fp=fp1, tgnn_fn=fn1, tgnn_tp=tp1,
        lr_tn=tn2,   lr_fp=fp2,   lr_fn=fn2,   lr_tp=tp2,
        rule_tn=tn3, rule_fp=fp3, rule_fn=fn3, rule_tp=tp3,
        fp_table=fp_table,
        ring_active=ring_active,
        tgnn_ring_det=tgnn_ring[1], tgnn_ring_rate=tgnn_ring[0],
        lr_ring_det=lr_ring[1],     lr_ring_rate=lr_ring[0],
        rule_ring_det=rule_ring[1], rule_ring_rate=rule_ring[0],
        top_rows=top_rows_str,
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(md, encoding="utf-8")
    print(f"[Results] Saved: {RESULTS_MD}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    print("=" * 66)
    print("PulseGuard-Z  --  Phase 2: Evaluation (Part A v2 dataset)")
    print("=" * 66)

    # ------------------------------------------------------------------ #
    # 1. Load everything                                                   #
    # ------------------------------------------------------------------ #
    txn, acct, id2idx, N, all_ids, acct_type_tensor = load_data()
    acct_type_arr = acct_type_tensor.numpy()

    ring_membership = load_ring_membership()
    all_snapshots   = build_all_snapshots(txn, N, id2idx, acct_type_tensor)
    model, ckpt     = load_model()

    # ------------------------------------------------------------------ #
    # 2. Split boundaries                                                  #
    # ------------------------------------------------------------------ #
    T_total         = len(all_snapshots)
    n_train         = int(T_total * TRAIN_FRAC)
    n_val           = int(T_total * VAL_FRAC)
    test_start_idx  = n_train + n_val

    t_min         = txn["timestamp"].min()
    window_td     = pd.Timedelta(minutes=WINDOW_MINUTES)
    t_train_start = t_min
    t_val_start   = t_min + n_train * window_td
    t_test_start  = t_min + test_start_idx * window_td
    t_data_end    = txn["timestamp"].max() + pd.Timedelta(seconds=1)

    print(f"\n[Split] Train: {n_train} | Val: {n_val} | Test: {T_total - test_start_idx} snapshots")
    print(f"        Train end: {t_val_start}")
    print(f"        Val   end: {t_test_start}")
    print(f"        Data  end: {txn['timestamp'].max()}")

    # ------------------------------------------------------------------ #
    # 3. Ground-truth labels                                               #
    # ------------------------------------------------------------------ #
    y_train_all = period_labels(txn, all_ids, t_train_start, t_val_start)
    y_val       = period_labels(txn, all_ids, t_val_start,   t_test_start)
    y_true      = period_labels(txn, all_ids, t_test_start,  t_data_end)

    n_fraud_accts_train = int(y_train_all.sum())
    n_fraud_accts_val   = int(y_val.sum())
    n_fraud_accts_test  = int(y_true.sum())
    print(f"\n[Labels] Train fraud accounts: {n_fraud_accts_train}")
    print(f"[Labels] Val   fraud accounts: {n_fraud_accts_val}")
    print(f"[Labels] Test  fraud accounts: {n_fraud_accts_test} (positive test examples)")

    # ------------------------------------------------------------------ #
    # 4. Rule-Baseline p99 from TRAINING data only                        #
    # ------------------------------------------------------------------ #
    train_txn = txn[txn["timestamp"] < t_val_start]
    p99_train = float(train_txn["amount"].quantile(0.99))
    print(f"\n[Baseline] p99 train-period threshold: {p99_train:.2f} INR")

    # ------------------------------------------------------------------ #
    # 5. LR features                                                       #
    # ------------------------------------------------------------------ #
    print("\n[LR] Building aggregated features ...")
    train_snaps = all_snapshots[:n_train]
    val_snaps   = all_snapshots[n_train: test_start_idx]
    test_snaps  = all_snapshots[test_start_idx:]

    X_train_lr = build_lr_features(train_snaps, N, acct_type_arr)
    y_train_lr = snap_period_labels(train_snaps, N)
    X_val_lr   = build_lr_features(val_snaps,   N, acct_type_arr)
    X_test_lr  = build_lr_features(test_snaps,  N, acct_type_arr)

    print(f"  X_train_lr shape: {X_train_lr.shape}  positives: {int(y_train_lr.sum())}")

    print("\n[LR] Training logistic regression ...")
    lr_model, lr_scaler, lr_C = train_lr(X_train_lr, y_train_lr, X_val_lr, y_val)

    # ------------------------------------------------------------------ #
    # 6. Validation-set scores for threshold selection                     #
    # ------------------------------------------------------------------ #
    print("\n[Val] Computing validation-set scores ...")

    # TGNN val inference
    tgnn_val_scores, _ = tgnn_inference(
        model, all_snapshots, n_train, test_start_idx, N
    )

    # LR val scores
    lr_val_scores = lr_model.predict_proba(lr_scaler.transform(X_val_lr))[:, 1]

    # Rule baseline val scores (binary)
    rule_val_scores = rule_baseline(
        txn, all_ids, N, p99_train, t_val_start, t_test_start
    )

    # ------------------------------------------------------------------ #
    # 7. Threshold selection (validation only)                             #
    # ------------------------------------------------------------------ #
    print("\n[Threshold] Selecting thresholds on validation set ...")
    tgnn_tau, tgnn_val_f1 = select_threshold(y_val, tgnn_val_scores, "TGNN")
    lr_tau,   lr_val_f1   = select_threshold(y_val, lr_val_scores,   "LR")
    # Rule baseline is binary → threshold = 0.5 by convention
    rule_tau = 0.5

    # ------------------------------------------------------------------ #
    # 8. Test-set inference (evaluated ONCE)                               #
    # ------------------------------------------------------------------ #
    print("\n[Inference] Running TGNN on test window ...")
    tgnn_test_scores, peak_features = tgnn_inference(
        model, all_snapshots, test_start_idx, T_total, N
    )

    print("[Inference] Running LR on test window ...")
    lr_test_scores = lr_model.predict_proba(lr_scaler.transform(X_test_lr))[:, 1]

    print("[Inference] Running rule baseline on test window ...")
    rule_test_scores = rule_baseline(
        txn, all_ids, N, p99_train, t_test_start, t_data_end
    )

    # ------------------------------------------------------------------ #
    # 9. Test metrics                                                       #
    # ------------------------------------------------------------------ #
    print("\n[Metrics] Test-set evaluation ...")
    tgnn_res = compute_metrics("TGNN",                y_true, tgnn_test_scores, tgnn_tau)
    lr_res   = compute_metrics("Logistic Regression", y_true, lr_test_scores,   lr_tau)
    rule_res = compute_metrics("Rule Baseline",       y_true, rule_test_scores, rule_tau)

    # ------------------------------------------------------------------ #
    # 10. FP breakdown                                                      #
    # ------------------------------------------------------------------ #
    tgnn_pred = (tgnn_test_scores >= tgnn_tau).astype(int)
    lr_pred   = (lr_test_scores   >= lr_tau).astype(int)
    rule_pred = (rule_test_scores  >= rule_tau).astype(int)

    tgnn_fp_by_type = fp_by_account_type(y_true, tgnn_pred, all_ids, acct)
    lr_fp_by_type   = fp_by_account_type(y_true, lr_pred,   all_ids, acct)
    rule_fp_by_type = fp_by_account_type(y_true, rule_pred, all_ids, acct)

    print(f"\n[FP] TGNN  FP by type: {tgnn_fp_by_type}")
    print(f"[FP] LR    FP by type: {lr_fp_by_type}")
    print(f"[FP] Rule  FP by type: {rule_fp_by_type}")

    # ------------------------------------------------------------------ #
    # 11. Ring detection                                                    #
    # ------------------------------------------------------------------ #
    print("\n[Ring] Computing ring detection rates ...")
    tgnn_ring_rate, tgnn_ring_det, ring_active = ring_detection_rate(
        tgnn_pred, all_ids, ring_membership, txn, t_test_start
    )
    lr_ring_rate, lr_ring_det, _   = ring_detection_rate(
        lr_pred,   all_ids, ring_membership, txn, t_test_start
    )
    rule_ring_rate, rule_ring_det, _ = ring_detection_rate(
        rule_pred, all_ids, ring_membership, txn, t_test_start
    )

    tgnn_ring = (tgnn_ring_rate, tgnn_ring_det)
    lr_ring   = (lr_ring_rate,   lr_ring_det)
    rule_ring = (rule_ring_rate, rule_ring_det)

    # ------------------------------------------------------------------ #
    # 12. Save scores CSV                                                   #
    # ------------------------------------------------------------------ #
    scores_df = save_scores_csv(
        all_ids, tgnn_test_scores, rule_test_scores,
        lr_test_scores, y_true, peak_features
    )

    # ------------------------------------------------------------------ #
    # 13. Plots                                                             #
    # ------------------------------------------------------------------ #
    vis_data = [
        ("TGNN",                 tgnn_test_scores,  "#0f62fe", "-"),
        ("Logistic Regression",  lr_test_scores,    "#24a148", "--"),
        ("Rule Baseline",        rule_test_scores,  "#da1e28", ":"),
    ]
    save_roc_chart(vis_data, y_true)
    save_pr_chart(vis_data, y_true)

    # ------------------------------------------------------------------ #
    # 14. Top rows for results.md                                           #
    # ------------------------------------------------------------------ #
    top_subset = scores_df.head(15)
    rows_str = []
    for _, r in top_subset.iterrows():
        rows_str.append(
            f"| {r['account_id']} | {r['tgnn_score']:.4f} | {r['true_label']} | "
            f"{r['top_signal_1']} | {r['top_signal_2']} | {r['top_signal_3']} |"
        )
    top_rows_str = "\n".join(rows_str)

    # ------------------------------------------------------------------ #
    # 15. Write results.md                                                  #
    # ------------------------------------------------------------------ #
    write_results_md(
        txn=txn, acct=acct,
        y_val=y_val, y_true=y_true,
        tgnn_res=tgnn_res, lr_res=lr_res, rule_res=rule_res,
        tgnn_tau=tgnn_tau, lr_tau=lr_tau,
        tgnn_val_f1=tgnn_val_f1, lr_val_f1=lr_val_f1,
        ckpt=ckpt, lr_C=lr_C,
        tgnn_fp_by_type=tgnn_fp_by_type,
        lr_fp_by_type=lr_fp_by_type,
        rule_fp_by_type=rule_fp_by_type,
        tgnn_ring=tgnn_ring, lr_ring=lr_ring, rule_ring=rule_ring,
        ring_active=ring_active,
        n_fraud_accts_train=n_fraud_accts_train,
        n_fraud_accts_val=n_fraud_accts_val,
        top_rows_str=top_rows_str,
    )

    print("=" * 66)
    print("EVALUATION COMPLETE")
    print(f"  Positive test examples : {n_fraud_accts_test}")
    print(f"  TGNN   ROC-AUC={tgnn_res['roc_auc']:.4f}  F1={tgnn_res['f1']:.4f}  "
          f"(tau={tgnn_tau:.2f})")
    print(f"  LR     ROC-AUC={lr_res['roc_auc']:.4f}  F1={lr_res['f1']:.4f}  "
          f"(tau={lr_tau:.2f})")
    print(f"  Rule   ROC-AUC={rule_res['roc_auc']:.4f}  F1={rule_res['f1']:.4f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
