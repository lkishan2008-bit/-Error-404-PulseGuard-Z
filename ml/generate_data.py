"""
==============================================================================
PulseGuard-Z  –  Phase 1: Synthetic G2P Transaction Data Generator
==============================================================================
Project   : PulseGuard-Z (IBM Z Hackathon, event code SSF2026)
Team      : Error-404
File      : ml/generate_data.py
Purpose   :
    Generates reproducible synthetic Government-to-Person (G2P) welfare
    transaction data that mimics a real-world payment network over three
    calendar days.  The output is used in subsequent phases for:
        - Training the Temporal Graph Neural Network (TGNN) fraud detector
        - Benchmarking baseline classifiers
        - Stress-testing the FastAPI inference backend

Simulation design
-----------------
Normal activity
    - ~2 000 accounts (beneficiaries, merchants, agents)
    - ~50 000 transactions over 72 hours
    - Realistic intra-day periodicity (low at night, peaks mid-morning /
      early evening)
    - Transaction types: welfare_payout, retail_purchase, p2p_transfer,
      agent_cash_out, utility_payment

Fraud patterns (2-4 % of transactions)
    - 8-12 planted rapid-transfer fraud rings:
        Each ring has 3-7 mule accounts that cycle funds in rapid bursts
        (< 2-minute hops) within a 10-30 minute window, simulating the
        "smurfing" / layering pattern common in G2P flash fraud.
    - 2 liquidity-drain bursts:
        A single compromised disbursing account drains its balance in a
        tight cluster of high-value cash-out transactions within minutes.

Outputs
-------
    data/transactions.csv   - ~50 000 rows
    data/accounts.csv       - ~2 000 rows
    docs/fraud_vs_normal_over_time.png
    docs/amount_distribution.png
    docs/transactions_per_account.png

Usage
-----
    python ml/generate_data.py

Fixed seed
----------
    All randomness is seeded with RANDOM_SEED = 42 for full reproducibility.

Dependencies (Phase 1 only)
---------------------------
    pandas, numpy, matplotlib  (see requirements.txt)
==============================================================================
"""

import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend - safe for servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED        = 42
N_ACCOUNTS         = 2_000
TARGET_TXN         = 50_000
SIM_START          = datetime(2026, 10, 1, 0, 0, 0)   # Day 1 00:00 UTC
SIM_DAYS           = 3
N_FRAUD_RINGS      = 10          # between 8-12; fixed for reproducibility
N_LIQUIDITY_BURSTS = 2
FRAUD_TARGET_FRAC  = 0.03        # ~3 %  (within 2-4 % spec)
# We guarantee exactly 1,500 fraud transactions:
#   10 rings x 130 hops  = 1,300 ring txns
#    2 drains x 100 cashouts = 200 drain txns
#   Total fraud = 1,500  ->  3.0 % of 50,000
RING_HOPS_EACH     = 130
DRAIN_CASHOUTS_EACH = 100

# Account-type distribution (must sum to 1.0)
ACCT_TYPE_WEIGHTS = {
    "beneficiary":  0.70,
    "merchant":     0.15,
    "agent":        0.10,
    "government":   0.05,
}

# Transaction types and their baseline weights for normal activity
TX_TYPE_WEIGHTS = {
    "welfare_payout":    0.18,
    "retail_purchase":   0.38,
    "p2p_transfer":      0.20,
    "agent_cash_out":    0.14,
    "utility_payment":   0.10,
}

# Typical amount ranges (min, max, mean) per tx type - log-normal sampled
TX_AMOUNT_PARAMS = {
    "welfare_payout":   (50,    1_500,  400),
    "retail_purchase":  (5,     500,    80),
    "p2p_transfer":     (10,    2_000,  200),
    "agent_cash_out":   (50,    3_000,  500),
    "utility_payment":  (20,    600,    120),
}

# Output paths (relative to repo root, regardless of CWD)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data"
DOCS_DIR  = REPO_ROOT / "docs"

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Step 1 - Generate accounts
# ---------------------------------------------------------------------------

def generate_accounts(n):
    """Return a DataFrame of `n` synthetic accounts."""
    types   = list(ACCT_TYPE_WEIGHTS.keys())
    weights = list(ACCT_TYPE_WEIGHTS.values())
    acct_types = rng.choice(types, size=n, p=weights)

    records = []
    for i in range(n):
        records.append({
            "account_id":        "ACC{:06d}".format(i),
            "account_type":      acct_types[i],
            "is_fraud_account":  0,     # will be updated later
        })

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Step 2 - Intra-day weighting (realistic periodicity)
# ---------------------------------------------------------------------------

def hour_weight(hour):
    """Return a relative activity weight for a given hour (0-23)."""
    profile = [
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2,   # 00-05
        0.5, 0.8, 1.0, 1.2, 1.5, 1.6,   # 06-11
        1.4, 1.1, 1.0, 0.9, 1.3, 1.6,   # 12-17
        1.5, 1.2, 0.9, 0.7, 0.5, 0.3,   # 18-23
    ]
    return profile[hour]


def sample_timestamps(n, start, days):
    """Sample `n` timestamps over `days` days with intra-day periodicity."""
    hours = range(days * 24)
    weights = np.array([hour_weight(h % 24) for h in hours], dtype=float)
    weights /= weights.sum()

    chosen_hours = rng.choice(len(hours), size=n, p=weights)
    offsets = rng.integers(0, 3600, size=n)
    timestamps = [
        start + timedelta(hours=int(h), seconds=int(s))
        for h, s in zip(chosen_hours, offsets)
    ]
    timestamps.sort()
    return timestamps


# ---------------------------------------------------------------------------
# Step 3 - Sample a log-normal amount for a given tx type
# ---------------------------------------------------------------------------

def sample_amount(tx_type, size=1):
    lo, hi, mean = TX_AMOUNT_PARAMS[tx_type]
    sigma = 0.8
    mu    = np.log(mean)
    vals  = rng.lognormal(mean=mu, sigma=sigma, size=size)
    vals  = np.clip(vals, lo, hi)
    return np.round(vals, 2)


# ---------------------------------------------------------------------------
# Step 4 - Generate normal transactions
# ---------------------------------------------------------------------------

def generate_normal_transactions(accounts, n, start, days):
    """Generate `n` normal (non-fraud) transactions."""

    account_ids = accounts["account_id"].values
    tx_types    = list(TX_TYPE_WEIGHTS.keys())
    tx_weights  = list(TX_TYPE_WEIGHTS.values())

    timestamps = sample_timestamps(n, start, days)
    chosen_tx  = rng.choice(tx_types, size=n, p=tx_weights)

    sender_idx   = rng.integers(0, len(account_ids), size=n)
    receiver_idx = rng.integers(0, len(account_ids), size=n)
    same_mask = sender_idx == receiver_idx
    receiver_idx[same_mask] = (receiver_idx[same_mask] + 1) % len(account_ids)

    amounts = np.concatenate([sample_amount(t, 1) for t in chosen_tx])

    records = []
    for i in range(n):
        records.append({
            "transaction_id": "TXN{:08d}".format(i),
            "timestamp":      timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id":      account_ids[sender_idx[i]],
            "receiver_id":    account_ids[receiver_idx[i]],
            "amount":         float(amounts[i]),
            "tx_type":        chosen_tx[i],
            "is_fraud":       0,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 5 - Plant rapid-transfer fraud rings
# ---------------------------------------------------------------------------

def plant_fraud_rings(accounts, txn_counter_start, start, days, n_rings):
    """
    Plant `n_rings` rapid-transfer fraud rings.

    Each ring:
    - Has 3-7 mule accounts (a subset of existing accounts)
    - Fires within a random 10-30 minute burst window
    - Produces exactly RING_HOPS_EACH hops for deterministic fraud count
    - Uses elevated amounts (800-3 000) to maximise velocity

    Total ring fraud txns = n_rings x RING_HOPS_EACH (e.g. 10 x 130 = 1,300)
    Returns (fraud_txns_df, list_of_fraud_account_ids).
    """
    account_ids    = accounts["account_id"].values
    fraud_txns     = []
    fraud_acct_ids = []
    ctr            = txn_counter_start

    for ring_idx in range(n_rings):
        ring_size  = int(rng.integers(3, 8))        # 3-7 mules per ring
        burst_min  = int(rng.integers(10, 31))      # 10-30 min window
        burst_len  = RING_HOPS_EACH                 # fixed for deterministic count

        max_offset = days * 24 * 3600 - burst_min * 60
        burst_start_offset = int(rng.integers(0, max_offset))
        burst_start = start + timedelta(seconds=burst_start_offset)

        mule_indices = rng.choice(len(account_ids), size=ring_size, replace=False)
        mules = account_ids[mule_indices].tolist()
        fraud_acct_ids.extend(mules)

        hop_offsets_s = np.sort(rng.integers(0, burst_min * 60, size=burst_len))

        for hop in range(burst_len):
            sender_i   = hop % ring_size
            receiver_i = (hop + 1) % ring_size
            hop_time   = burst_start + timedelta(seconds=int(hop_offsets_s[hop]))
            amount     = float(np.round(rng.uniform(800, 3_000), 2))

            fraud_txns.append({
                "transaction_id": "TXN{:08d}".format(ctr),
                "timestamp":      hop_time.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id":      mules[sender_i],
                "receiver_id":    mules[receiver_i],
                "amount":         amount,
                "tx_type":        "p2p_transfer",
                "is_fraud":       1,
            })
            ctr += 1

    return pd.DataFrame(fraud_txns), list(set(fraud_acct_ids))


# ---------------------------------------------------------------------------
# Step 6 - Plant liquidity-drain bursts
# ---------------------------------------------------------------------------

def plant_liquidity_drain(accounts, txn_counter_start, start, days, n_bursts):
    """
    Plant `n_bursts` liquidity-drain events.

    Each burst:
    - One compromised disbursing account sends exactly DRAIN_CASHOUTS_EACH
      cash-outs to distinct receivers within a 20-40 minute tight window
    - High amounts (1 000-5 000)

    Total drain fraud txns = n_bursts x DRAIN_CASHOUTS_EACH (e.g. 2 x 100 = 200)
    """
    account_ids    = accounts["account_id"].values
    fraud_txns     = []
    fraud_acct_ids = []
    ctr            = txn_counter_start

    for _ in range(n_bursts):
        n_cashouts = DRAIN_CASHOUTS_EACH           # fixed for deterministic count
        window_min = int(rng.integers(20, 41))     # 20-40 min window

        max_offset  = days * 24 * 3600 - window_min * 60
        burst_start = start + timedelta(seconds=int(rng.integers(0, max_offset)))

        src_idx = int(rng.integers(0, len(account_ids)))
        src     = account_ids[src_idx]
        fraud_acct_ids.append(src)

        other_indices = [i for i in range(len(account_ids)) if i != src_idx]
        recv_indices  = rng.choice(other_indices, size=n_cashouts, replace=False)
        receivers     = account_ids[recv_indices].tolist()
        offsets_s     = np.sort(rng.integers(0, window_min * 60, size=n_cashouts))

        for j in range(n_cashouts):
            t      = burst_start + timedelta(seconds=int(offsets_s[j]))
            amount = float(np.round(rng.uniform(1_000, 5_000), 2))
            fraud_txns.append({
                "transaction_id": "TXN{:08d}".format(ctr),
                "timestamp":      t.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id":      src,
                "receiver_id":    receivers[j],
                "amount":         amount,
                "tx_type":        "agent_cash_out",
                "is_fraud":       1,
            })
            ctr += 1

    return pd.DataFrame(fraud_txns), list(set(fraud_acct_ids))


# ---------------------------------------------------------------------------
# Step 7 - Assemble full dataset
# ---------------------------------------------------------------------------

def assemble_dataset():
    print("=" * 66)
    print("PulseGuard-Z  --  Phase 1 Data Generator")
    print("Seed: {}  |  Start: {}  |  Days: {}".format(
        RANDOM_SEED, SIM_START.date(), SIM_DAYS))
    print("=" * 66)

    # Step 1: accounts
    print("\n[1/5] Generating accounts ...")
    accounts = generate_accounts(N_ACCOUNTS)
    print("      Created {:,} accounts.".format(len(accounts)))

    # Step 2: fraud first (so we know exact fraud count before filling normals)
    print("\n[2/5] Planting {} rapid-transfer fraud rings ...".format(N_FRAUD_RINGS))
    ring_df, ring_accts = plant_fraud_rings(
        accounts,
        txn_counter_start=0,
        start=SIM_START,
        days=SIM_DAYS,
        n_rings=N_FRAUD_RINGS,
    )
    print("      Ring transactions: {:,}  |  Mule accounts: {}".format(
        len(ring_df), len(ring_accts)))

    print("\n[3/5] Planting {} liquidity-drain bursts ...".format(N_LIQUIDITY_BURSTS))
    drain_df, drain_accts = plant_liquidity_drain(
        accounts,
        txn_counter_start=len(ring_df),
        start=SIM_START,
        days=SIM_DAYS,
        n_bursts=N_LIQUIDITY_BURSTS,
    )
    print("      Drain transactions: {:,}  |  Source accounts: {}".format(
        len(drain_df), len(drain_accts)))

    # Step 3: fill normals to reach TARGET_TXN total
    n_fraud  = len(ring_df) + len(drain_df)
    n_normal = max(TARGET_TXN - n_fraud, 0)
    print("\n[4/5] Generating {:,} normal transactions (total target: {:,}) ...".format(
        n_normal, TARGET_TXN))
    normal_df = generate_normal_transactions(accounts, n_normal, SIM_START, SIM_DAYS)

    all_txns = pd.concat([normal_df, ring_df, drain_df], ignore_index=True)
    all_txns["timestamp"] = pd.to_datetime(all_txns["timestamp"])
    all_txns.sort_values("timestamp", inplace=True)
    all_txns.reset_index(drop=True, inplace=True)
    all_txns["transaction_id"] = ["TXN{:08d}".format(i) for i in range(len(all_txns))]

    all_fraud_accts = set(ring_accts) | set(drain_accts)
    accounts["is_fraud_account"] = accounts["account_id"].isin(all_fraud_accts).astype(int)

    return accounts, all_txns, all_fraud_accts


# ---------------------------------------------------------------------------
# Step 8 - Save CSVs
# ---------------------------------------------------------------------------

def save_csvs(accounts, transactions):
    print("\n[5/5] Saving CSVs ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    acct_path = DATA_DIR / "accounts.csv"
    txn_path  = DATA_DIR / "transactions.csv"

    accounts.to_csv(acct_path, index=False)
    transactions.to_csv(txn_path, index=False)

    print("      Saved: {}".format(acct_path))
    print("      Saved: {}".format(txn_path))
    return txn_path, acct_path


# ---------------------------------------------------------------------------
# Step 9 - Generate diagnostic charts
# ---------------------------------------------------------------------------

GREY_NORMAL = "#9E9E9E"
RED_FRAUD   = "#E53935"


def _common_fig():
    """Return a white-background figure and axis."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.tick_params(colors="#444444")
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")
    ax.title.set_color("#222222")
    return fig, ax


def chart_fraud_vs_normal_over_time(df, out_path):
    """Hourly transaction count -- normal (grey) vs fraud (red) stacked bars."""
    df2       = df.copy()
    df2["hour"] = df2["timestamp"].dt.floor("h")

    normal   = df2[df2["is_fraud"] == 0].groupby("hour").size().rename("normal")
    fraud    = df2[df2["is_fraud"] == 1].groupby("hour").size().rename("fraud")
    combined = pd.concat([normal, fraud], axis=1).fillna(0)

    fig, ax = _common_fig()
    ax.bar(combined.index, combined["normal"], label="Normal", color=GREY_NORMAL,
           width=0.04, align="center")
    ax.bar(combined.index, combined["fraud"],  label="Fraud",  color=RED_FRAUD,
           width=0.04, align="center", bottom=combined["normal"])

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=30)

    ax.set_title("Hourly Transactions: Normal vs Fraud", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Transaction Count")
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("      Chart saved: {}".format(out_path))


def chart_amount_distribution(df, out_path):
    """Overlapping histograms of transaction amounts -- normal vs fraud."""
    normal_amounts = df[df["is_fraud"] == 0]["amount"].clip(upper=5_000)
    fraud_amounts  = df[df["is_fraud"] == 1]["amount"].clip(upper=5_000)

    fig, ax = _common_fig()
    ax.hist(normal_amounts, bins=80, color=GREY_NORMAL, alpha=0.75,
            label="Normal", density=True, edgecolor="none")
    ax.hist(fraud_amounts,  bins=80, color=RED_FRAUD,   alpha=0.85,
            label="Fraud",  density=True, edgecolor="none")

    ax.set_title("Transaction Amount Distribution (clipped at 5,000)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Amount (USD-equivalent)")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("      Chart saved: {}".format(out_path))


def chart_transactions_per_account(df, out_path):
    """Histogram of how many transactions each sender account has."""
    normal_df = df[df["is_fraud"] == 0]
    fraud_df  = df[df["is_fraud"] == 1]

    normal_counts = normal_df.groupby("sender_id").size().clip(upper=200)
    fraud_counts  = fraud_df.groupby("sender_id").size().clip(upper=200)

    fig, ax = _common_fig()
    ax.hist(normal_counts, bins=60, color=GREY_NORMAL, alpha=0.75,
            label="Normal senders", density=True, edgecolor="none")
    ax.hist(fraud_counts,  bins=60, color=RED_FRAUD,   alpha=0.85,
            label="Fraud senders",  density=True, edgecolor="none")

    ax.set_title("Transactions per Sender Account (clipped at 200)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Outgoing Transaction Count")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("      Chart saved: {}".format(out_path))


def generate_charts(df):
    print("\n[Charts] Generating diagnostic charts ...")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart_fraud_vs_normal_over_time(df, DOCS_DIR / "fraud_vs_normal_over_time.png")
    chart_amount_distribution(df,       DOCS_DIR / "amount_distribution.png")
    chart_transactions_per_account(df,  DOCS_DIR / "transactions_per_account.png")


# ---------------------------------------------------------------------------
# Step 10 - Verify and print summary
# ---------------------------------------------------------------------------

def print_summary(accounts, transactions, txn_path, acct_path):
    n_accts = len(accounts)
    n_txns  = len(transactions)
    n_fraud = int(transactions["is_fraud"].sum())
    frac    = n_fraud / n_txns * 100
    t_min   = transactions["timestamp"].min()
    t_max   = transactions["timestamp"].max()

    print("\n" + "=" * 66)
    print("PHASE 1 VERIFICATION SUMMARY")
    print("=" * 66)
    print("  Account count      : {:,}".format(n_accts))
    print("  Transaction count  : {:,}".format(n_txns))
    print("  Fraud count        : {:,}".format(n_fraud))
    print("  Fraud percentage   : {:.2f} %".format(frac))
    print("  Time range         : {}  ->  {}".format(t_min, t_max))
    print("  transactions.csv   : {}".format(txn_path))
    print("  accounts.csv       : {}".format(acct_path))
    print("  Charts             : {}".format(DOCS_DIR))
    print("=" * 66)

    assert n_accts >= 1_800,  "Too few accounts: {}".format(n_accts)
    assert n_txns  >= 40_000, "Too few transactions: {}".format(n_txns)
    assert 2.0 <= frac <= 4.0, "Fraud fraction out of 2-4 % range: {:.2f} %".format(frac)
    print("\n  All spec checks passed.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    accounts, transactions, _ = assemble_dataset()
    txn_path, acct_path = save_csvs(accounts, transactions)
    generate_charts(transactions)
    print_summary(accounts, transactions, txn_path, acct_path)
