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
in plain PyTorch (no PyTorch Geometric or DGL) to satisfy the hackathon's
CPU-demo constraint and ensure full transparency and auditability without
external C++ graph extensions.

Pipeline
--------
1. Load data/transactions.csv and data/accounts.csv.
2. Vectorized 30-minute snapshot construction:
     - Edge list and node feature extraction per snapshot.
     - Node features (8 dims):
         0: out_count
         1: in_count
         2: log_out_amount = log1p(out_amount)
         3: log_in_amount  = log1p(in_amount)
         4: forwarding_ratio = out_amount / (in_amount + 1e-9)
         5: unique_receivers
         6: unique_senders
         7: amount_std (standard deviation of transaction amounts)
3. Normalized sparse adjacency matrix per snapshot:
     - Undirected edge aggregation + self-loops
     - Symmetric degree normalization: A_norm = D^{-1/2} A D^{-1/2}
     - Stored as torch.sparse_coo_tensor to avoid dense N x N allocations.
4. Model Architecture (TGNN):
     - 2 Graph Convolutional Layers with ReLU activation and Dropout:
         H' = ReLU(A_norm @ H @ W + b)
     - GRU over T=6 snapshots (3-hour temporal lookback) per node.
     - Linear head with Sigmoid activation to produce node fraud probability.
5. Chronological Split & Supervision:
     - Chronological train (70%) / validation (15%) / test (15%) split.
     - Target label aligned to the current snapshot of each sequence.
     - Balanced positive-weighted BCE loss to address severe class imbalance.
6. Checkpoint:
     - Saves trained model weights and metadata to ml/tgnn_model.pt.

Usage
-----
    python ml/train_tgnn.py
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

# Silence benign PyTorch sparse invariant warning on CPU
torch.sparse.check_sparse_tensor_invariants.disable()

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
N_SNAPSHOTS_SEQ = 6           # GRU look-back (3 hours)
NODE_FEAT_DIM   = 8
GCN_HIDDEN      = 32
GCN_LAYERS      = 2
GRU_HIDDEN      = 64
DROPOUT         = 0.30
LR              = 3e-3
MAX_EPOCHS      = 25
PATIENCE        = 6
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15        # test = remainder (15%)

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

    n_fraud_txns = int((txn["is_fraud"] == 1).sum())
    n_fraud_accs = int((acct["is_fraud_account"] == 1).sum())
    print(f"       Accounts: {N:,}  |  Fraud accounts: {n_fraud_accs:,}")
    print(f"       Transactions: {len(txn):,}  |  Fraud txns: {n_fraud_txns:,}")

    return txn, acct, id2idx, N, all_ids


# ---------------------------------------------------------------------------
# 2.  Vectorized Feature Engineering per Snapshot
# ---------------------------------------------------------------------------

def build_snapshot_features(src, dst, amounts, N):
    """
    Compute 8 node features for accounts in a snapshot using vectorized NumPy operations.
    Feat 0: out_count
    Feat 1: in_count
    Feat 2: log_out_amount
    Feat 3: log_in_amount
    Feat 4: forwarding_ratio = out_amt / (in_amt + 1e-9)
    Feat 5: unique_receivers
    Feat 6: unique_senders
    Feat 7: amount_std
    """
    feats = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)
    if len(src) == 0:
        return feats

    # 0 & 1: out_count and in_count
    np.add.at(feats[:, 0], src, 1.0)
    np.add.at(feats[:, 1], dst, 1.0)

    # 2 & 3: out_amount and in_amount
    np.add.at(feats[:, 2], src, amounts)
    np.add.at(feats[:, 3], dst, amounts)

    # 4: forwarding_ratio
    feats[:, 4] = feats[:, 2] / (feats[:, 3] + 1e-9)

    # 5: unique_receivers
    edges_out = np.column_stack([src, dst])
    uniq_out = np.unique(edges_out, axis=0)
    np.add.at(feats[:, 5], uniq_out[:, 0], 1.0)

    # 6: unique_senders
    edges_in = np.column_stack([dst, src])
    uniq_in = np.unique(edges_in, axis=0)
    np.add.at(feats[:, 6], uniq_in[:, 0], 1.0)

    # 7: amount_std
    all_accts   = np.concatenate([src, dst])
    all_amounts = np.concatenate([amounts, amounts])
    amt_series  = pd.Series(all_amounts, index=all_accts)
    amt_std     = amt_series.groupby(level=0).std(ddof=0).fillna(0)
    feats[amt_std.index.values, 7] = amt_std.values.astype(np.float32)

    # Log-transform out and in amounts
    feats[:, 2] = np.log1p(feats[:, 2])
    feats[:, 3] = np.log1p(feats[:, 3])

    return feats


# ---------------------------------------------------------------------------
# 3.  Sparse Adjacency (COO) with Symmetric Degree Normalization
# ---------------------------------------------------------------------------

def build_sparse_adj(src, dst, N):
    """
    Build a normalized adjacency matrix as a torch.sparse_coo_tensor.
    Includes undirected edge aggregation, self-loops, and D^{-1/2} A D^{-1/2} scaling.
    """
    if len(src) == 0:
        idx    = torch.arange(N)
        edge_i = torch.stack([idx, idx])
        vals   = torch.ones(N)
        return torch.sparse_coo_tensor(edge_i, vals, (N, N))

    s_all  = np.concatenate([src, dst, np.arange(N, dtype=np.int32)])
    d_all  = np.concatenate([dst, src, np.arange(N, dtype=np.int32)])
    edge_i = torch.tensor(np.stack([s_all, d_all]), dtype=torch.long)
    vals   = torch.ones(len(s_all), dtype=torch.float32)

    A = torch.sparse_coo_tensor(edge_i, vals, (N, N)).coalesce()

    deg     = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))

    i_idx     = A.indices()[0]
    j_idx     = A.indices()[1]
    norm_vals = A.values() * deg_inv[i_idx] * deg_inv[j_idx]

    return torch.sparse_coo_tensor(A.indices(), norm_vals, (N, N)).coalesce()


# ---------------------------------------------------------------------------
# 4.  Build All Graph Snapshots (Exact Vectorized Window Assignment)
# ---------------------------------------------------------------------------

def build_all_snapshots(txn, N, id2idx):
    print("[Snapshots] Building 30-minute graph snapshots ...")
    t_min     = txn["timestamp"].min()
    window_td = pd.Timedelta(minutes=WINDOW_MINUTES)

    # Vectorized window assignment avoiding any timestamp unit mismatch
    window_indices = ((txn["timestamp"] - t_min) // window_td).astype(int).values
    n_windows      = int(window_indices.max()) + 1

    src_arr    = txn["sender_idx"].values
    dst_arr    = txn["receiver_idx"].values
    amount_arr = txn["amount"].values.astype(np.float32)
    fraud_arr  = txn["is_fraud"].values.astype(np.int32)

    snapshots = []
    fraud_label_count = 0

    for w in range(n_windows):
        mask = (window_indices == w)
        s = src_arr[mask].astype(np.int32)
        d = dst_arr[mask].astype(np.int32)
        a = amount_arr[mask]
        f = fraud_arr[mask]

        feats = build_snapshot_features(s, d, a, N)
        adj   = build_sparse_adj(s, d, N)

        lbl = torch.zeros(N, dtype=torch.float32)
        if len(f) > 0 and f.sum() > 0:
            lbl[s[f == 1]] = 1.0
            lbl[d[f == 1]] = 1.0
            fraud_label_count += int(lbl.sum().item())

        snapshots.append((torch.tensor(feats), adj, lbl))

    print(f"       Total snapshots: {len(snapshots)}  |  Total positive node-snapshots: {fraud_label_count}")
    return snapshots


# ---------------------------------------------------------------------------
# 5.  Model Definition
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Normalized graph convolution: H' = ReLU(A_norm @ H @ W + b)"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h, adj_norm):
        agg = torch.sparse.mm(adj_norm, h)
        return F.relu(self.W(agg))


class TGNN(nn.Module):
    """
    Temporal GNN: 2 GCN layers + GRU over T snapshots + linear classification head.
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
        return h

    def forward(self, feats_seq, adj_seq):
        embs = [self.encode_snapshot(f, a) for f, a in zip(feats_seq, adj_seq)]
        stacked = torch.stack(embs, dim=0)       # [T, N, gcn_hidden]
        gru_out, _ = self.gru(stacked)           # [T, N, gru_hidden]
        last = gru_out[-1]                        # [N, gru_hidden]
        return torch.sigmoid(self.head(last).squeeze(-1))  # [N]


# ---------------------------------------------------------------------------
# 6.  Chronological Split Helpers
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
# 7.  Training Helpers
# ---------------------------------------------------------------------------

def pos_weight_from(snap_list):
    total_pos = sum(s[2].sum().item() for s in snap_list)
    total_neg = sum((1 - s[2]).sum().item() for s in snap_list)
    if total_pos == 0:
        return torch.tensor(1.0)
    # Balanced square-root weighting: penalizes false negatives effectively
    # without overwhelming precision on extreme class imbalance.
    raw_ratio = total_neg / total_pos
    balanced_pw = min(raw_ratio ** 0.5, 50.0)
    return torch.tensor(balanced_pw, dtype=torch.float32)


def run_epoch(model, snap_list, optimizer, pw, train_mode):
    model.train(train_mode)
    criterion = nn.BCELoss(reduction="none")
    total_loss = 0.0
    n_seqs     = 0

    for i in range(len(snap_list) - N_SNAPSHOTS_SEQ + 1):
        seq_f  = [snap_list[j][0] for j in range(i, i + N_SNAPSHOTS_SEQ)]
        seq_a  = [snap_list[j][1] for j in range(i, i + N_SNAPSHOTS_SEQ)]
        # Supervise on the current window (end of the temporal look-back sequence)
        labels = snap_list[i + N_SNAPSHOTS_SEQ - 1][2]

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
# 8.  Main Train Function
# ---------------------------------------------------------------------------

def train(snapshots):
    print("\n[Train] Setting up model ...")
    train_s, val_s, test_s = chronological_split(snapshots)
    pw = pos_weight_from(train_s)
    print(f"       pos_weight: {pw.item():.2f}")

    model     = TGNN(NODE_FEAT_DIM, GCN_HIDDEN, GRU_HIDDEN, GCN_LAYERS, DROPOUT)
    optimizer = Adam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None
    train_losses   = []
    val_losses     = []

    print(f"[Train] max {MAX_EPOCHS} epochs | patience {PATIENCE} ...")
    for epoch in range(1, MAX_EPOCHS + 1):
        t0      = time.time()
        tr_l    = run_epoch(model, train_s, optimizer, pw, train_mode=True)
        vl_l    = run_epoch(model, val_s,   optimizer, pw, train_mode=False)
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

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "state_dict":       model.state_dict(),
        "feat_dim":         NODE_FEAT_DIM,
        "gcn_hidden":       GCN_HIDDEN,
        "gru_hidden":       GRU_HIDDEN,
        "gcn_layers":       GCN_LAYERS,
        "dropout":          DROPOUT,
        "window_minutes":   WINDOW_MINUTES,
        "n_snapshots_seq":  N_SNAPSHOTS_SEQ,
        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "best_val_loss":    best_val_loss,
        "pos_weight":       pw.item(),
    }, MODEL_PATH)

    print(f"\n[Saved] Model weights saved to: {MODEL_PATH}")
    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    print("=" * 66)
    print("PulseGuard-Z  --  Phase 2: Temporal GNN Training")
    print(f"Seed: {SEED}  |  Window: {WINDOW_MINUTES} min  |  Seq: {N_SNAPSHOTS_SEQ} steps")
    print("=" * 66)

    txn, acct, id2idx, N, all_ids = load_data()
    snapshots = build_all_snapshots(txn, N, id2idx)
    model, tr_l, vl_l = train(snapshots)

    print("=" * 66)
    print("TRAINING COMPLETE")
    print(f"  Best val loss : {min(vl_l):.4f}")
    print(f"  Final train loss : {tr_l[-1]:.4f}")
    print(f"  Model saved to: {MODEL_PATH}")
    print("=" * 66)


if __name__ == "__main__":
    main()
