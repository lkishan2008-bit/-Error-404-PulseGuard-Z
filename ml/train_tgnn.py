"""
==============================================================================
PulseGuard-Z  -  Phase 2: Temporal GNN Trainer
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : ml/train_tgnn.py

Architecture Overview
---------------------
We implement a lightweight Temporal Graph Neural Network (TGNN) entirely
in plain PyTorch (no PyG / DGL) to satisfy the hackathon's CPU-demo
constraint and make the code auditable without extra C++ extensions.

Pipeline
--------
1. Load data/transactions.csv  +  data/accounts.csv
2. Feature engineering per account per 30-minute snapshot (vectorised):
     - out_count, in_count, out_amount, in_amount (log-transformed)
     - forwarding_ratio  = out_amount / (in_amount + 1e-9)  [smurfing signal]
     - unique_receivers, unique_senders
     - amount_std  (stddev of all amounts touching this account in window)
3. Graph construction per snapshot:
     - Nodes  = all 2 000 accounts
     - Edges  = transactions in the 30-min window (undirected aggregate)
     - Adjacency: COO sparse -> normalised D^{-1/2} A D^{-1/2} stored as
       a (src, dst, val) edge list to avoid dense N*N matrix in memory
4. Model:
     GraphConvLayer  (2 layers, ReLU, dropout=0.3)
         - sparse message-passing: h' = sigma(D^{-1/2} A D^{-1/2} h W)
         - implemented via torch.sparse for efficiency
     GRU   (over T=6 snapshots, hidden=64, per node)
     Linear head -> sigmoid -> fraud probability per node
5. Training:
     - Chronological split: 70% train / 15% val / 15% test
     - Weighted BCE to handle 97/3 class imbalance
     - Adam, lr=1e-3, up to 30 epochs, early stopping patience=5
6. Artefacts:
     - ml/tgnn_model.pt   (state_dict + meta)

Limitations
-----------
- Synthetic data: planted fraud patterns, metrics may be optimistic.
- CPU-only small model; no GPU acceleration in this phase.
- GRU sees 6 snapshots (3 hours); longer history would help.
- First-order graph neighbourhood only.

Usage
-----
    python ml/train_tgnn.py

Dependencies
------------
    torch>=2.2.0, numpy>=1.26.0, pandas>=2.2.0, scikit-learn>=1.4.0
==============================================================================
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
ML_DIR     = REPO_ROOT / "ml"

TXN_CSV    = DATA_DIR / "transactions.csv"
ACCT_CSV   = DATA_DIR / "accounts.csv"
MODEL_PATH = ML_DIR   / "tgnn_model.pt"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
WINDOW_MINUTES  = 30
N_SNAPSHOTS_SEQ = 6           # GRU look-back
NODE_FEAT_DIM   = 8
GCN_HIDDEN      = 32
GCN_LAYERS      = 2
GRU_HIDDEN      = 64
DROPOUT         = 0.30
LR              = 1e-3
MAX_EPOCHS      = 30
PATIENCE        = 5
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15        # test = remainder

# ---------------------------------------------------------------------------
# 1.  Data loading
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

    fraud_senders   = set(txn.loc[txn["is_fraud"]==1, "sender_id"])
    fraud_receivers = set(txn.loc[txn["is_fraud"]==1, "receiver_id"])
    fraud_accts     = fraud_senders | fraud_receivers
    acct_labels     = torch.zeros(N, dtype=torch.float32)
    for aid, i in id2idx.items():
        if aid in fraud_accts:
            acct_labels[i] = 1.0

    print(f"       Accounts: {N:,}  |  Fraud accounts: {int(acct_labels.sum()):,}")
    print(f"       Transactions: {len(txn):,}  |  Fraud txns: {int((txn['is_fraud']==1).sum()):,}")
    return txn, acct, id2idx, N, acct_labels

# ---------------------------------------------------------------------------
# 2.  Vectorised feature engineering
# ---------------------------------------------------------------------------

def build_snapshot_features(src, dst, amounts, N):
    """
    All-numpy vectorised feature computation for one snapshot window.

    Parameters
    ----------
    src, dst : int32 arrays of sender / receiver indices
    amounts  : float64 array of transaction amounts
    N        : total number of accounts

    Returns
    -------
    feats : np.ndarray of shape (N, NODE_FEAT_DIM) float32
    """
    feats = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)
    if len(src) == 0:
        return feats

    # Counts and sums with np.add.at
    np.add.at(feats[:, 0], src, 1)          # out_count
    np.add.at(feats[:, 1], dst, 1)          # in_count
    np.add.at(feats[:, 2], src, amounts)    # out_amount
    np.add.at(feats[:, 3], dst, amounts)    # in_amount

    # forwarding_ratio (capped at 20)
    feats[:, 4] = np.clip(feats[:, 2] / (feats[:, 3] + 1e-9), 0, 20)

    # unique_receivers per sender, unique_senders per receiver
    # Use pandas groupby for unique counts
    if len(src) > 0:
        src_s = pd.Series(dst, index=src)
        uniq_r = src_s.groupby(level=0).nunique()
        feats[uniq_r.index.values, 5] = uniq_r.values.astype(np.float32)

        dst_s = pd.Series(src, index=dst)
        uniq_s = dst_s.groupby(level=0).nunique()
        feats[uniq_s.index.values, 6] = uniq_s.values.astype(np.float32)

    # amount_std per account
    # Pool amounts for each account (as sender or receiver)
    all_accts  = np.concatenate([src, dst])
    all_amounts = np.concatenate([amounts, amounts])
    # Use pandas for grouped std
    amt_series = pd.Series(all_amounts, index=all_accts)
    amt_std    = amt_series.groupby(level=0).std(ddof=0).fillna(0)
    feats[amt_std.index.values, 7] = amt_std.values.astype(np.float32)

    # log-transform out/in amounts
    feats[:, 2] = np.log1p(feats[:, 2])
    feats[:, 3] = np.log1p(feats[:, 3])

    return feats

# ---------------------------------------------------------------------------
# 3.  Sparse adjacency (COO) for efficient graph conv
# ---------------------------------------------------------------------------

def build_sparse_adj(src, dst, N):
    """
    Build a normalised adjacency as a torch.sparse_coo_tensor.
    Undirected: add both (s->r) and (r->s).
    Add self-loops, then D^{-1/2} A D^{-1/2} normalisation.

    Returns torch.sparse_coo_tensor of shape (N, N).
    """
    if len(src) == 0:
        # Identity (self-loops only)
        idx  = torch.arange(N)
        edge_i = torch.stack([idx, idx])
        vals   = torch.ones(N)
        return torch.sparse_coo_tensor(edge_i, vals, (N, N))

    # Combine directed edges + self-loops (undirected)
    s_all = np.concatenate([src, dst, np.arange(N, dtype=np.int32)])
    d_all = np.concatenate([dst, src, np.arange(N, dtype=np.int32)])
    edge_i = torch.tensor(np.stack([s_all, d_all]), dtype=torch.long)
    vals   = torch.ones(len(s_all), dtype=torch.float32)

    # Coalesce (sum duplicates) to get adjacency weights
    A = torch.sparse_coo_tensor(edge_i, vals, (N, N)).coalesce()

    # Degree normalisation
    deg     = torch.sparse.sum(A, dim=1).to_dense()          # [N]
    deg_inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))

    # D^{-1/2} A D^{-1/2}: scale each edge (i,j) by deg_inv[i]*deg_inv[j]
    i_idx   = A.indices()[0]
    j_idx   = A.indices()[1]
    norm_vals = A.values() * deg_inv[i_idx] * deg_inv[j_idx]

    return torch.sparse_coo_tensor(A.indices(), norm_vals, (N, N)).coalesce()

# ---------------------------------------------------------------------------
# 4.  Build ALL snapshots
# ---------------------------------------------------------------------------

def build_all_snapshots(txn, N, id2idx):
    print("[Snapshots] Building 30-minute graph snapshots ...")
    t_min     = txn["timestamp"].min()
    t_max     = txn["timestamp"].max()
    window_td = pd.Timedelta(minutes=WINDOW_MINUTES)

    src_arr     = txn["sender_idx"].values
    dst_arr     = txn["receiver_idx"].values
    amount_arr  = txn["amount"].values
    time_arr    = txn["timestamp"].values  # numpy datetime64
    fraud_arr   = txn["is_fraud"].values

    snapshots = []
    cur = t_min
    window_ns = int(window_td.total_seconds() * 1e9)   # nanoseconds
    cur_ns    = int(pd.Timestamp(cur).value)

    while cur <= t_max:
        end_ns = cur_ns + window_ns
        mask   = (time_arr.astype(np.int64) >= cur_ns) & \
                 (time_arr.astype(np.int64) <  end_ns)

        s = src_arr[mask].astype(np.int32)
        d = dst_arr[mask].astype(np.int32)
        a = amount_arr[mask]
        f = fraud_arr[mask]

        feats = build_snapshot_features(s, d, a, N)
        adj   = build_sparse_adj(s, d, N)

        # Per-snapshot node fraud label
        lbl = torch.zeros(N, dtype=torch.float32)
        if f.sum() > 0:
            lbl[s[f == 1]] = 1.0
            lbl[d[f == 1]] = 1.0

        snapshots.append((torch.tensor(feats), adj, lbl))
        cur_ns += window_ns
        cur    += window_td

    print(f"       Total snapshots: {len(snapshots)}")
    return snapshots

# ---------------------------------------------------------------------------
# 5.  Model definition
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Normalised graph convolution: H' = relu(A_norm @ H @ W + b)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h, adj_norm):
        # adj_norm: sparse [N, N]
        agg = torch.sparse.mm(adj_norm, h)  # [N, in_dim]
        return F.relu(self.W(agg))


class TGNN(nn.Module):
    """
    Temporal GNN: two GCN layers + GRU over T snapshots + linear head.

    Forward:
        feats_seq : list[T] of tensors [N, feat_dim]
        adj_seq   : list[T] of sparse tensors [N, N]
    Returns:
        probs : [N] fraud probabilities
    """
    def __init__(self, feat_dim, gcn_hidden, gru_hidden,
                 n_gcn_layers, dropout):
        super().__init__()
        layers = []
        in_d = feat_dim
        for _ in range(n_gcn_layers):
            layers.append(GraphConvLayer(in_d, gcn_hidden))
            in_d = gcn_hidden
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout    = nn.Dropout(dropout)
        self.gru        = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden,
                                  num_layers=1, batch_first=False)
        self.head       = nn.Linear(gru_hidden, 1)

    def encode_snapshot(self, feats, adj):
        h = feats
        for layer in self.gcn_layers:
            h = layer(h, adj)
            h = self.dropout(h)
        return h   # [N, gcn_hidden]

    def forward(self, feats_seq, adj_seq):
        embs = [self.encode_snapshot(f, a) for f, a in zip(feats_seq, adj_seq)]
        stacked = torch.stack(embs, dim=0)      # [T, N, gcn_hidden]
        gru_out, _ = self.gru(stacked)          # [T, N, gru_hidden]
        last = gru_out[-1]                       # [N, gru_hidden]
        return torch.sigmoid(self.head(last).squeeze(-1))  # [N]

# ---------------------------------------------------------------------------
# 6.  Chronological split helpers
# ---------------------------------------------------------------------------

def chronological_split(snapshots):
    T       = len(snapshots)
    n_train = int(T * TRAIN_FRAC)
    n_val   = int(T * VAL_FRAC)
    train_s = snapshots[:n_train]
    val_s   = snapshots[n_train: n_train + n_val]
    test_s  = snapshots[n_train + n_val:]
    print(f"[Split] Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)} snapshots")
    return train_s, val_s, test_s

# ---------------------------------------------------------------------------
# 7.  Training helpers
# ---------------------------------------------------------------------------

def pos_weight_from(snap_list):
    total_pos = sum(s[2].sum().item() for s in snap_list)
    total_neg = sum((1 - s[2]).sum().item() for s in snap_list)
    if total_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(total_neg / total_pos, dtype=torch.float32)


def run_epoch(model, snap_list, optimizer, pw, train_mode):
    model.train(train_mode)
    criterion = nn.BCELoss(reduction="none")
    total_loss = 0.0
    n_seqs     = 0

    for i in range(len(snap_list) - N_SNAPSHOTS_SEQ):
        seq_f  = [snap_list[j][0] for j in range(i, i + N_SNAPSHOTS_SEQ)]
        seq_a  = [snap_list[j][1] for j in range(i, i + N_SNAPSHOTS_SEQ)]
        labels = snap_list[i + N_SNAPSHOTS_SEQ][2]

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            probs = model(seq_f, seq_a)
            w     = torch.where(labels == 1, pw, torch.ones_like(labels))
            loss  = (criterion(probs, labels) * w).mean()
            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_seqs     += 1

    return total_loss / max(n_seqs, 1)

# ---------------------------------------------------------------------------
# 8.  Main train function
# ---------------------------------------------------------------------------

def train(snapshots):
    print("\n[Train] Setting up model ...")
    train_s, val_s, test_s = chronological_split(snapshots)
    pw = pos_weight_from(train_s)
    print(f"       pos_weight (neg/pos): {pw.item():.2f}")

    model     = TGNN(NODE_FEAT_DIM, GCN_HIDDEN, GRU_HIDDEN, GCN_LAYERS, DROPOUT)
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    train_losses   = []
    val_losses     = []

    print(f"[Train] max {MAX_EPOCHS} epochs | patience {PATIENCE} ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0     = time.time()
        tr_l   = run_epoch(model, train_s, optimizer, pw, train_mode=True)
        vl_l   = run_epoch(model, val_s,   optimizer, pw, train_mode=False)
        elapsed = time.time() - t0
        train_losses.append(tr_l)
        val_losses.append(vl_l)
        print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  train={tr_l:.4f}  val={vl_l:.4f}  ({elapsed:.1f}s)")

        if vl_l < best_val_loss:
            best_val_loss  = vl_l
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    torch.save({
        "state_dict":      model.state_dict(),
        "feat_dim":        NODE_FEAT_DIM,
        "gcn_hidden":      GCN_HIDDEN,
        "gru_hidden":      GRU_HIDDEN,
        "gcn_layers":      GCN_LAYERS,
        "dropout":         DROPOUT,
        "window_minutes":  WINDOW_MINUTES,
        "n_snapshots_seq": N_SNAPSHOTS_SEQ,
        "train_losses":    train_losses,
        "val_losses":      val_losses,
        "best_val_loss":   best_val_loss,
        "pos_weight":      float(pw.item()),
    }, MODEL_PATH)
    print(f"\n[Train] Saved: {MODEL_PATH}  (best_val_loss={best_val_loss:.4f})")
    return model, train_s, val_s, test_s, pw

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 66)
    print("PulseGuard-Z  --  Phase 2: TGNN Training")
    print("=" * 66)
    txn, acct, id2idx, N, acct_labels = load_data()
    snapshots = build_all_snapshots(txn, N, id2idx)
    model, train_s, val_s, test_s, pw = train(snapshots)
    print("\n[Done] Run ml/evaluate.py for metrics and scores.")
