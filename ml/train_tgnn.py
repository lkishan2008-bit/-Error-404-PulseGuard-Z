"""
==============================================================================
PulseGuard-Z  -  Phase 2: Temporal GNN Trainer (Part B2 Improved)
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : ml/train_tgnn.py

Architecture Overview
---------------------
Improved Temporal Graph Neural Network (TGNN) implemented in plain PyTorch:
1. Leakage-safe feature scaling using StandardScaler fitted strictly on
   training-period behavioral features.
2. Temporal graph signal via transaction volume/frequency edge weighting:
   A_uv = log1p(amount) + 1.0, with symmetric degree normalization:
   A_norm = D^{-1/2} A D^{-1/2}
3. Concatenation architecture:
   Combines final GRU hidden state (temporal context) with target snapshot
   raw node features (instantaneous state) into a 2-layer MLP classification head:
   H_comb = [H_GRU || X_raw] -> Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> Sigmoid
4. Validation-only 4-configuration grid tuning for (lr, pos_weight) with
   early stopping, running under 10 minutes on CPU.
5. Preserves exact untouched test period (snapshots 121..143).
==============================================================================
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
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
# Hyper-parameters & Split Boundaries
# ---------------------------------------------------------------------------
WINDOW_MINUTES  = 30
N_SNAPSHOTS_SEQ = 6           # GRU look-back (3 hours)

# Feature dimensions
BEHAV_FEAT_DIM  = 8           # Behavioural features computed per snapshot
ACCT_TYPES      = ["agent", "beneficiary", "government", "merchant"]
NODE_FEAT_DIM   = BEHAV_FEAT_DIM + len(ACCT_TYPES)  # 8 + 4 = 12

GCN_HIDDEN      = 32
GCN_LAYERS      = 2
GRU_HIDDEN      = 64
DROPOUT         = 0.30
MAX_EPOCHS      = 12
PATIENCE        = 3

# Exact chronological split counts:
# Total 144 snapshots (30-min windows)
# Train: snapshots 0..85  (86 snapshots, 59.7%)
# Val:   snapshots 86..120 (35 snapshots, 24.3%) -> 108 fraud accounts (>= 50 required!)
# Test:  snapshots 121..143 (23 snapshots, 16.0%) -> 102 fraud accounts (strictly untouched)
N_TRAIN_SNAPS   = 86
N_VAL_SNAPS     = 35
TRAIN_FRAC      = 86 / 144
VAL_FRAC        = 35 / 144


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

    acct_type_tensor = build_account_type_matrix(acct, all_ids)
    print(f"       Account-type one-hot: {list(acct_type_tensor.shape)}  (types: {ACCT_TYPES})")

    return txn, acct, id2idx, N, all_ids, acct_type_tensor


# ---------------------------------------------------------------------------
# 1b.  Static Account-Type One-Hot Matrix
# ---------------------------------------------------------------------------

def build_account_type_matrix(acct, all_ids):
    """
    Build a static [N, len(ACCT_TYPES)] float32 tensor ordered by all_ids.
    """
    type2col = {t: i for i, t in enumerate(ACCT_TYPES)}
    id2type  = dict(zip(acct["account_id"], acct["account_type"]))
    N        = len(all_ids)
    mat      = np.zeros((N, len(ACCT_TYPES)), dtype=np.float32)
    for i, aid in enumerate(all_ids):
        atype = id2type.get(aid, ACCT_TYPES[0])  # fallback: agent
        col   = type2col.get(atype, 0)
        mat[i, col] = 1.0
    return torch.tensor(mat, dtype=torch.float32)  # [N, 4]


# ---------------------------------------------------------------------------
# 2.  Vectorized Feature Engineering per Snapshot
# ---------------------------------------------------------------------------

def build_snapshot_features(src, dst, amounts, N):
    """
    Compute 8 raw behavioral node features for accounts in a snapshot.
    """
    feats = np.zeros((N, BEHAV_FEAT_DIM), dtype=np.float32)
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
# 3.  Sparse Adjacency (COO) with Volume Weighting & Symmetric Normalization
# ---------------------------------------------------------------------------

def build_sparse_adj(src, dst, amounts, N):
    """
    Build a normalized adjacency matrix as a torch.sparse_coo_tensor.
    Incorporates transaction volume/frequency edge weights and self-loops:
    A_norm = D^{-1/2} A D^{-1/2}
    """
    if len(src) == 0:
        idx    = torch.arange(N)
        edge_i = torch.stack([idx, idx])
        vals   = torch.ones(N)
        return torch.sparse_coo_tensor(edge_i, vals, (N, N)).coalesce()

    s_all  = np.concatenate([src, dst, np.arange(N, dtype=np.int32)])
    d_all  = np.concatenate([dst, src, np.arange(N, dtype=np.int32)])
    # Volume/frequency signal: log1p(amount) + 1.0 baseline weight for transactions, 1.0 for self-loops
    amt_weight = np.log1p(amounts).astype(np.float32) + 1.0
    w_all = np.concatenate([amt_weight, amt_weight, np.ones(N, dtype=np.float32)])

    edge_i = torch.tensor(np.stack([s_all, d_all]), dtype=torch.long)
    vals   = torch.tensor(w_all, dtype=torch.float32)

    A = torch.sparse_coo_tensor(edge_i, vals, (N, N)).coalesce()

    deg     = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))

    i_idx     = A.indices()[0]
    j_idx     = A.indices()[1]
    norm_vals = A.values() * deg_inv[i_idx] * deg_inv[j_idx]

    return torch.sparse_coo_tensor(A.indices(), norm_vals, (N, N)).coalesce()


# ---------------------------------------------------------------------------
# 4.  Build All Graph Snapshots with Leakage-Safe StandardScaler
# ---------------------------------------------------------------------------

def build_all_snapshots(txn, N, id2idx, acct_type_tensor, scaler=None):
    """
    Build 30-minute graph snapshots.
    If scaler is None, fit StandardScaler on the training snapshots only.
    """
    print("[Snapshots] Building 30-minute graph snapshots ...")
    t_min     = txn["timestamp"].min()
    window_td = pd.Timedelta(minutes=WINDOW_MINUTES)

    window_indices = ((txn["timestamp"] - t_min) // window_td).astype(int).values
    n_windows      = int(window_indices.max()) + 1

    src_arr    = txn["sender_idx"].values
    dst_arr    = txn["receiver_idx"].values
    amount_arr = txn["amount"].values.astype(np.float32)
    fraud_arr  = txn["is_fraud"].values.astype(np.int32)

    raw_behav_list = []
    adj_list = []
    lbl_list = []
    fraud_label_count = 0

    for w in range(n_windows):
        mask = (window_indices == w)
        s = src_arr[mask].astype(np.int32)
        d = dst_arr[mask].astype(np.int32)
        a = amount_arr[mask]
        f = fraud_arr[mask]

        behav = build_snapshot_features(s, d, a, N)
        raw_behav_list.append(behav)

        adj = build_sparse_adj(s, d, a, N)
        adj_list.append(adj)

        lbl = torch.zeros(N, dtype=torch.float32)
        if len(f) > 0 and f.sum() > 0:
            lbl[s[f == 1]] = 1.0
            lbl[d[f == 1]] = 1.0
            fraud_label_count += int(lbl.sum().item())
        lbl_list.append(lbl)

    # Fit scaler strictly on training snapshots to prevent leakage
    if scaler is None:
        train_count = min(N_TRAIN_SNAPS, n_windows)
        train_behav_concat = np.vstack(raw_behav_list[:train_count])
        scaler = StandardScaler()
        scaler.fit(train_behav_concat)
        print(f"       [Scaler] Fitted StandardScaler on first {train_count} training snapshots.")

    snapshots = []
    for w in range(n_windows):
        scaled_behav = scaler.transform(raw_behav_list[w])
        behav_t = torch.tensor(scaled_behav, dtype=torch.float32)
        feats = torch.cat([behav_t, acct_type_tensor], dim=1)
        snapshots.append((feats, adj_list[w], lbl_list[w]))

    print(f"       Total snapshots: {len(snapshots)}  |  Total positive node-snapshots: {fraud_label_count}")
    return snapshots, scaler


# ---------------------------------------------------------------------------
# 5.  Model Definition (Concatenation Architecture)
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
    Improved Temporal GNN:
    2 GCN layers + GRU over T snapshots.
    Concatenates final GRU hidden state with raw scaled node features of the target window.
    2-layer MLP classification head with LayerNorm and Dropout.
    """
    def __init__(self, feat_dim, gcn_hidden, gru_hidden,
                 n_gcn_layers, dropout):
        super().__init__()
        self.feat_dim   = feat_dim
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden
        self.gcn_layers_count = n_gcn_layers
        self.dropout_val = dropout

        layers = []
        in_d = feat_dim
        for _ in range(n_gcn_layers):
            layers.append(GraphConvLayer(in_d, gcn_hidden))
            in_d = gcn_hidden
        self.gcn_layers = nn.ModuleList(layers)
        self.dropout    = nn.Dropout(dropout)
        self.gru        = nn.GRU(input_size=gcn_hidden, hidden_size=gru_hidden,
                                 num_layers=1, batch_first=False)

        # Concatenate final GRU hidden state [N, gru_hidden] + raw node features [N, feat_dim]
        comb_dim = gru_hidden + feat_dim
        self.head = nn.Sequential(
            nn.Linear(comb_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

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
        last_gru = gru_out[-1]                   # [N, gru_hidden]
        target_raw = feats_seq[-1]               # [N, feat_dim]
        comb = torch.cat([last_gru, target_raw], dim=-1)  # [N, gru_hidden + feat_dim]
        logits = self.head(comb).squeeze(-1)     # [N]
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# 6.  Chronological Split Helpers
# ---------------------------------------------------------------------------

def chronological_split(snapshots):
    T = len(snapshots)
    n_train = min(N_TRAIN_SNAPS, T)
    n_val   = min(N_VAL_SNAPS, max(0, T - n_train))
    train_s = snapshots[:n_train]
    val_s   = snapshots[n_train: n_train + n_val]
    test_s  = snapshots[n_train + n_val:]
    print(f"[Split] Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)} snapshots")
    return train_s, val_s, test_s


# ---------------------------------------------------------------------------
# 7.  Training & Validation Helpers
# ---------------------------------------------------------------------------

def run_epoch(model, snap_list, optimizer, pw, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    n_seqs     = 0
    pos_w = torch.tensor(pw, dtype=torch.float32)

    for i in range(len(snap_list) - N_SNAPSHOTS_SEQ + 1):
        seq_f  = [snap_list[j][0] for j in range(i, i + N_SNAPSHOTS_SEQ)]
        seq_a  = [snap_list[j][1] for j in range(i, i + N_SNAPSHOTS_SEQ)]
        labels = snap_list[i + N_SNAPSHOTS_SEQ - 1][2]

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            probs = model(seq_f, seq_a)
            eps = 1e-7
            probs_c = torch.clamp(probs, eps, 1.0 - eps)
            w = torch.where(labels == 1, pos_w, torch.ones_like(labels))
            loss = -(w * (labels * torch.log(probs_c) + (1.0 - labels) * torch.log(1.0 - probs_c))).mean()
            if train_mode:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_seqs     += 1

    return total_loss / max(n_seqs, 1)


def evaluate_val_f1(model, val_s, y_val, N):
    model.eval()
    max_scores = np.zeros(N, dtype=np.float64)
    with torch.no_grad():
        for i in range(len(val_s) - N_SNAPSHOTS_SEQ + 1):
            seq_f = [val_s[j][0] for j in range(i, i + N_SNAPSHOTS_SEQ)]
            seq_a = [val_s[j][1] for j in range(i, i + N_SNAPSHOTS_SEQ)]
            probs = model(seq_f, seq_a).cpu().numpy()
            max_scores = np.maximum(max_scores, probs)

    best_tau = 0.50
    best_f1  = 0.0
    for tau in np.arange(0.02, 0.98, 0.02):
        pred = (max_scores >= tau).astype(int)
        f1   = f1_score(y_val, pred, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)

    return best_f1, best_tau, max_scores


# ---------------------------------------------------------------------------
# 8.  Validation-Only Grid Tuning (4 Configurations)
# ---------------------------------------------------------------------------

CONFIG_GRID = [
    {"lr": 1e-3, "pos_weight": 12.0},
    {"lr": 1e-3, "pos_weight": 18.0},
    {"lr": 3e-3, "pos_weight": 12.0},
    {"lr": 3e-3, "pos_weight": 18.0},
]


def train_and_tune(snapshots, scaler, N):
    train_s, val_s, test_s = chronological_split(snapshots)

    # Compute validation ground truth per account from validation snapshot labels
    val_labels_t = torch.zeros(N, dtype=torch.float32)
    for _, _, lbl in val_s:
        val_labels_t = torch.clamp(val_labels_t + lbl, max=1.0)
    y_val = val_labels_t.numpy()
    print(f"       Validation ground truth fraud accounts: {int(y_val.sum())} (out of {N})")

    best_overall_f1    = -1.0
    best_overall_cfg   = None
    best_overall_state = None
    best_overall_loss  = float("inf")
    tuning_history     = []

    print("\n[Tune] Running 4-configuration validation tuning (max 12 epochs each) ...")
    start_all = time.time()

    for idx, cfg in enumerate(CONFIG_GRID, 1):
        lr_val = cfg["lr"]
        pw_val = cfg["pos_weight"]
        print(f"\n--- Config {idx}/4: lr={lr_val:.4f}, pos_weight={pw_val:.1f} ---")

        # Fixed seed per configuration for reproducibility
        torch.manual_seed(SEED + idx)
        model = TGNN(NODE_FEAT_DIM, GCN_HIDDEN, GRU_HIDDEN, GCN_LAYERS, DROPOUT)
        optimizer = Adam(model.parameters(), lr=lr_val, weight_decay=1e-4)

        best_cfg_f1    = -1.0
        best_cfg_loss  = float("inf")
        best_cfg_state = None
        patience_count = 0
        train_losses   = []
        val_losses     = []

        for epoch in range(1, MAX_EPOCHS + 1):
            t0 = time.time()
            tr_l = run_epoch(model, train_s, optimizer, pw_val, train_mode=True)
            vl_l = run_epoch(model, val_s,   optimizer, pw_val, train_mode=False)
            val_f1, val_tau, _ = evaluate_val_f1(model, val_s, y_val, N)
            elapsed = time.time() - t0

            train_losses.append(tr_l)
            val_losses.append(vl_l)
            print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  tr_loss={tr_l:.4f}  vl_loss={vl_l:.4f}  val_F1={val_f1:.4f} (tau={val_tau:.2f})  ({elapsed:.1f}s)")

            # Track best model state within this config based on validation F1
            if val_f1 > best_cfg_f1 or (val_f1 == best_cfg_f1 and vl_l < best_cfg_loss):
                best_cfg_f1    = val_f1
                best_cfg_loss  = vl_l
                best_cfg_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stopping config {idx} at epoch {epoch}.")
                    break

        cfg_summary = {
            "config_id": idx,
            "lr": lr_val,
            "pos_weight": pw_val,
            "best_val_f1": best_cfg_f1,
            "best_val_loss": best_cfg_loss,
            "epochs_run": len(train_losses),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        tuning_history.append(cfg_summary)
        print(f"  Config {idx} finished: best_val_F1={best_cfg_f1:.4f}, best_val_loss={best_cfg_loss:.4f}")

        if best_cfg_f1 > best_overall_f1 or (best_cfg_f1 == best_overall_f1 and best_cfg_loss < best_overall_loss):
            best_overall_f1    = best_cfg_f1
            best_overall_loss  = best_cfg_loss
            best_overall_cfg   = cfg
            best_overall_state = best_cfg_state

    total_tuning_time = time.time() - start_all
    print(f"\n[Tune] Tuning complete in {total_tuning_time:.1f}s ({total_tuning_time/60:.2f} min).")
    print(f"[Tune] Best config: lr={best_overall_cfg['lr']}, pos_weight={best_overall_cfg['pos_weight']} (Val F1={best_overall_f1:.4f})")

    # Load and save best overall checkpoint
    best_model = TGNN(NODE_FEAT_DIM, GCN_HIDDEN, GRU_HIDDEN, GCN_LAYERS, DROPOUT)
    best_model.load_state_dict(best_overall_state)

    torch.save({
        "state_dict":         best_model.state_dict(),
        "feat_dim":           NODE_FEAT_DIM,
        "behav_feat_dim":     BEHAV_FEAT_DIM,
        "acct_types":         ACCT_TYPES,
        "gcn_hidden":         GCN_HIDDEN,
        "gru_hidden":         GRU_HIDDEN,
        "gcn_layers":         GCN_LAYERS,
        "dropout":            DROPOUT,
        "window_minutes":     WINDOW_MINUTES,
        "n_snapshots_seq":    N_SNAPSHOTS_SEQ,
        "best_val_loss":      best_overall_loss,
        "best_val_f1":        best_overall_f1,
        "pos_weight":         best_overall_cfg["pos_weight"],
        "lr":                 best_overall_cfg["lr"],
        "scaler_mean":        scaler.mean_,
        "scaler_scale":       scaler.scale_,
        "tuning_history":     tuning_history,
        "total_tuning_time":  total_tuning_time,
        "n_train_snaps":      N_TRAIN_SNAPS,
        "n_val_snaps":        N_VAL_SNAPS,
    }, MODEL_PATH)

    print(f"\n[Saved] Best TGNN model checkpoint saved to: {MODEL_PATH}")
    return best_model, tuning_history, best_overall_cfg


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    print("=" * 66)
    print("PulseGuard-Z  --  Phase 2: Improved TGNN Training & Validation Tuning")
    print(f"Seed: {SEED}  |  Window: {WINDOW_MINUTES} min  |  Seq: {N_SNAPSHOTS_SEQ} steps")
    print(f"Split: Train={N_TRAIN_SNAPS} snaps | Val={N_VAL_SNAPS} snaps | Test={144 - N_TRAIN_SNAPS - N_VAL_SNAPS} snaps")
    print("=" * 66)

    txn, acct, id2idx, N, all_ids, acct_type_tensor = load_data()
    snapshots, scaler = build_all_snapshots(txn, N, id2idx, acct_type_tensor)
    model, tuning_history, best_cfg = train_and_tune(snapshots, scaler, N)

    print("=" * 66)
    print("TRAINING & TUNING COMPLETE")
    print(f"  Best Val F1   : {max(h['best_val_f1'] for h in tuning_history):.4f}")
    print(f"  Selected Cfg  : {best_cfg}")
    print(f"  Model saved to: {MODEL_PATH}")
    print("=" * 66)


if __name__ == "__main__":
    main()
