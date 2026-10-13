import random
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# PulseGuard-Z Phase 1 Data Generator (v2)
# IBM Z Hackathon SSF2026 - Team Error-404
# v2 vs v1:
#   - 30 rings spread deterministically across 72h (was 10 clustered)
#   - 4 drain bursts spread across 72h (was 2)
#   - INR amounts (was USD-equivalent labels)
#   - Per-window fraud labels (node flagged only in windows it transacts)
#   - Split stats + ring placement table printed to stdout

RANDOM_SEED         = 42
N_ACCOUNTS          = 2_000
TARGET_TXN          = 50_000
SIM_START           = datetime(2026, 10, 1, 0, 0, 0)
SIM_DAYS            = 3
WINDOW_MINUTES      = 30

N_FRAUD_RINGS       = 30
RING_HOPS_EACH      = 50
N_LIQUIDITY_BURSTS  = 4
DRAIN_CASHOUTS_EACH = 75

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

ACCT_TYPE_WEIGHTS = {
    'beneficiary': 0.70,
    'merchant':    0.15,
    'agent':       0.10,
    'government':  0.05,
}

TX_TYPE_WEIGHTS = {
    'welfare_payout':  0.18,
    'retail_purchase': 0.38,
    'p2p_transfer':    0.20,
    'agent_cash_out':  0.14,
    'utility_payment': 0.10,
}

# INR amounts (min, max, mean) - log-normal sampled
TX_AMOUNT_PARAMS = {
    'welfare_payout':  (500,    10_000, 2_500),
    'retail_purchase': (20,      3_000,   250),
    'p2p_transfer':    (50,     15_000, 1_000),
    'agent_cash_out':  (200,    20_000, 2_500),
    'utility_payment': (100,     5_000,   600),
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / 'data'
DOCS_DIR  = REPO_ROOT / 'docs'

rng = np.random.default_rng(RANDOM_SEED)
import random as _random
_random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_accounts(n):
    types   = list(ACCT_TYPE_WEIGHTS.keys())
    weights = list(ACCT_TYPE_WEIGHTS.values())
    acct_types = rng.choice(types, size=n, p=weights)
    records = []
    for i in range(n):
        records.append({
            'account_id':       f'ACC{i:06d}',
            'account_type':     acct_types[i],
            'is_fraud_account': 0,
        })
    return pd.DataFrame(records)


def hour_weight(hour):
    profile = [
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2,
        0.5, 0.8, 1.0, 1.2, 1.5, 1.6,
        1.4, 1.1, 1.0, 0.9, 1.3, 1.6,
        1.5, 1.2, 0.9, 0.7, 0.5, 0.3,
    ]
    return profile[hour]


def sample_timestamps(n, start, days):
    hours   = range(days * 24)
    weights = np.array([hour_weight(h % 24) for h in hours], dtype=float)
    weights /= weights.sum()
    chosen_hours = rng.choice(len(hours), size=n, p=weights)
    offsets      = rng.integers(0, 3600, size=n)
    timestamps   = [
        start + timedelta(hours=int(h), seconds=int(s))
        for h, s in zip(chosen_hours, offsets)
    ]
    timestamps.sort()
    return timestamps


def sample_amount(tx_type, size=1):
    lo, hi, mean = TX_AMOUNT_PARAMS[tx_type]
    sigma = 0.8
    mu    = np.log(mean)
    vals  = rng.lognormal(mean=mu, sigma=sigma, size=size)
    vals  = np.clip(vals, lo, hi)
    return np.round(vals, 2)


def generate_normal_transactions(accounts, n, start, days):
    account_ids = accounts['account_id'].values
    tx_types    = list(TX_TYPE_WEIGHTS.keys())
    tx_weights  = list(TX_TYPE_WEIGHTS.values())
    timestamps  = sample_timestamps(n, start, days)
    chosen_tx   = rng.choice(tx_types, size=n, p=tx_weights)
    sender_idx   = rng.integers(0, len(account_ids), size=n)
    receiver_idx = rng.integers(0, len(account_ids), size=n)
    same_mask    = sender_idx == receiver_idx
    receiver_idx[same_mask] = (receiver_idx[same_mask] + 1) % len(account_ids)
    amounts = np.concatenate([sample_amount(t, 1) for t in chosen_tx])
    records = []
    for i in range(n):
        records.append({
            'transaction_id': f'TXN{i:08d}',
            'timestamp':      timestamps[i].strftime('%Y-%m-%d %H:%M:%S'),
            'sender_id':      account_ids[sender_idx[i]],
            'receiver_id':    account_ids[receiver_idx[i]],
            'amount':         float(amounts[i]),
            'tx_type':        chosen_tx[i],
            'is_fraud':       0,
        })
    return pd.DataFrame(records)


def plant_fraud_rings(accounts, txn_counter_start, start, days, n_rings):
    # Spread rings across 72h using equal-width slots with jitter.
    # 30 rings x 50 hops = 1500 fraud txns.  INR amounts 5000-25000.
    total_seconds = days * 24 * 3600
    slot_seconds  = total_seconds / n_rings
    account_ids    = accounts['account_id'].values
    fraud_txns     = []
    fraud_acct_ids = []
    ring_info      = []
    ctr            = txn_counter_start
    for ring_idx in range(n_rings):
        ring_size = int(rng.integers(4, 8))
        burst_min = int(rng.integers(10, 31))
        slot_start   = ring_idx * slot_seconds
        max_jitter   = max(0.0, slot_seconds - burst_min * 60 - 10)
        jitter       = int(rng.integers(0, max(1, int(max_jitter))))
        burst_offset = int(slot_start) + jitter
        burst_offset = min(burst_offset, total_seconds - burst_min * 60 - 60)
        burst_offset = max(0, burst_offset)
        burst_start  = start + timedelta(seconds=burst_offset)
        mule_indices = rng.choice(len(account_ids), size=ring_size, replace=False)
        mules        = account_ids[mule_indices].tolist()
        fraud_acct_ids.extend(mules)
        ring_info.append({'ring_idx': ring_idx, 'burst_start': burst_start,
                          'n_mules': ring_size, 'mules': mules})
        hop_offsets_s = np.sort(rng.integers(0, burst_min * 60, size=RING_HOPS_EACH))
        for hop in range(RING_HOPS_EACH):
            sender_i   = hop % ring_size
            receiver_i = (hop + 1) % ring_size
            hop_time   = burst_start + timedelta(seconds=int(hop_offsets_s[hop]))
            amount     = float(np.round(rng.uniform(5_000, 25_000), 2))
            fraud_txns.append({
                'transaction_id': f'TXN{ctr:08d}',
                'timestamp':      hop_time.strftime('%Y-%m-%d %H:%M:%S'),
                'sender_id':      mules[sender_i],
                'receiver_id':    mules[receiver_i],
                'amount':         amount,
                'tx_type':        'p2p_transfer',
                'is_fraud':       1,
            })
            ctr += 1
    return pd.DataFrame(fraud_txns), list(set(fraud_acct_ids)), ring_info


def plant_liquidity_drain(accounts, txn_counter_start, start, days, n_bursts):
    # 4 bursts x 75 cashouts = 300 fraud txns.  INR amounts 8000-40000.
    total_seconds = days * 24 * 3600
    slot_seconds  = total_seconds / n_bursts
    account_ids    = accounts['account_id'].values
    fraud_txns     = []
    fraud_acct_ids = []
    ctr            = txn_counter_start
    for burst_i in range(n_bursts):
        window_min   = int(rng.integers(20, 41))
        slot_start   = burst_i * slot_seconds
        max_jitter   = max(0.0, slot_seconds - window_min * 60 - 10)
        jitter       = int(rng.integers(0, max(1, int(max_jitter))))
        burst_offset = int(slot_start) + jitter
        burst_offset = min(burst_offset, total_seconds - window_min * 60 - 60)
        burst_offset = max(0, burst_offset)
        burst_start  = start + timedelta(seconds=burst_offset)
        src_idx = int(rng.integers(0, len(account_ids)))
        src     = account_ids[src_idx]
        fraud_acct_ids.append(src)
        other_indices = [i for i in range(len(account_ids)) if i != src_idx]
        recv_indices  = rng.choice(other_indices, size=DRAIN_CASHOUTS_EACH, replace=False)
        receivers     = account_ids[recv_indices].tolist()
        offsets_s     = np.sort(rng.integers(0, window_min * 60, size=DRAIN_CASHOUTS_EACH))
        for j in range(DRAIN_CASHOUTS_EACH):
            t      = burst_start + timedelta(seconds=int(offsets_s[j]))
            amount = float(np.round(rng.uniform(8_000, 40_000), 2))
            fraud_txns.append({
                'transaction_id': f'TXN{ctr:08d}',
                'timestamp':      t.strftime('%Y-%m-%d %H:%M:%S'),
                'sender_id':      src,
                'receiver_id':    receivers[j],
                'amount':         amount,
                'tx_type':        'agent_cash_out',
                'is_fraud':       1,
            })
            ctr += 1
    return pd.DataFrame(fraud_txns), list(set(fraud_acct_ids))


def assemble_dataset():
    print('=' * 70)
    print('PulseGuard-Z  --  Phase 1 Data Generator  (v2, INR, 30 rings)')
    print(f'Seed: {RANDOM_SEED}  |  Start: {SIM_START.date()}  |  Days: {SIM_DAYS}')
    print('=' * 70)

    print('\n[1/5] Generating accounts ...')
    accounts = generate_accounts(N_ACCOUNTS)
    print(f'      Created {len(accounts):,} accounts.')

    print(f'\n[2/5] Planting {N_FRAUD_RINGS} fraud rings across 72 h ...')
    ring_df, ring_accts, ring_info = plant_fraud_rings(
        accounts, txn_counter_start=0,
        start=SIM_START, days=SIM_DAYS, n_rings=N_FRAUD_RINGS,
    )
    print(f'      Ring txns: {len(ring_df):,}  |  Unique mule accounts: {len(ring_accts)}')

    print(f'\n[3/5] Planting {N_LIQUIDITY_BURSTS} drain bursts ...')
    drain_df, drain_accts = plant_liquidity_drain(
        accounts, txn_counter_start=len(ring_df),
        start=SIM_START, days=SIM_DAYS, n_bursts=N_LIQUIDITY_BURSTS,
    )
    print(f'      Drain txns: {len(drain_df):,}  |  Source accounts: {len(drain_accts)}')

    n_fraud  = len(ring_df) + len(drain_df)
    n_normal = max(TARGET_TXN - n_fraud, 0)
    print(f'\n[4/5] Generating {n_normal:,} normal transactions ...')
    normal_df = generate_normal_transactions(accounts, n_normal, SIM_START, SIM_DAYS)

    all_txns = pd.concat([normal_df, ring_df, drain_df], ignore_index=True)
    all_txns['timestamp'] = pd.to_datetime(all_txns['timestamp'])
    all_txns.sort_values('timestamp', inplace=True)
    all_txns.reset_index(drop=True, inplace=True)
    all_txns['transaction_id'] = [f'TXN{i:08d}' for i in range(len(all_txns))]

    all_fraud_accts = set(ring_accts) | set(drain_accts)
    accounts['is_fraud_account'] = accounts['account_id'].isin(all_fraud_accts).astype(int)

    return accounts, all_txns, all_fraud_accts, ring_info


def save_csvs(accounts, transactions):
    print('\n[5/5] Saving CSVs ...')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    acct_path = DATA_DIR / 'accounts.csv'
    txn_path  = DATA_DIR / 'transactions.csv'
    accounts.to_csv(acct_path, index=False)
    transactions.to_csv(txn_path, index=False)
    print(f'      Saved: {acct_path}')
    print(f'      Saved: {txn_path}')
    return txn_path, acct_path


GREY_NORMAL = '#9E9E9E'
RED_FRAUD   = '#E53935'


def _common_fig():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
    ax.tick_params(colors='#444444')
    ax.xaxis.label.set_color('#444444')
    ax.yaxis.label.set_color('#444444')
    ax.title.set_color('#222222')
    return fig, ax


def chart_fraud_vs_normal_over_time(df, out_path):
    df2 = df.copy()
    df2['hour'] = df2['timestamp'].dt.floor('h')
    normal   = df2[df2['is_fraud'] == 0].groupby('hour').size().rename('normal')
    fraud    = df2[df2['is_fraud'] == 1].groupby('hour').size().rename('fraud')
    combined = pd.concat([normal, fraud], axis=1).fillna(0)
    fig, ax = _common_fig()
    ax.bar(combined.index, combined['normal'], label='Normal',
           color=GREY_NORMAL, width=0.04, align='center')
    ax.bar(combined.index, combined['fraud'],  label='Fraud',
           color=RED_FRAUD, width=0.04, align='center', bottom=combined['normal'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %Hh'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=30, ha='right')
    ax.set_title('Transactions per Hour - Normal vs Fraud (INR)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (IST)')
    ax.set_ylabel('Transaction Count')
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'      Chart saved: {out_path}')


def chart_amount_distribution(df, out_path):
    clip_inr       = 30_000
    normal_amounts = df[df['is_fraud'] == 0]['amount'].clip(upper=clip_inr)
    fraud_amounts  = df[df['is_fraud'] == 1]['amount'].clip(upper=clip_inr)
    fig, ax = _common_fig()
    ax.hist(normal_amounts, bins=80, color=GREY_NORMAL, alpha=0.75,
            label='Normal', density=True, edgecolor='none')
    ax.hist(fraud_amounts,  bins=80, color=RED_FRAUD,   alpha=0.85,
            label='Fraud',  density=True, edgecolor='none')
    ax.set_title(f'Transaction Amount Distribution - clipped at Rs {clip_inr}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Amount (INR)')
    ax.set_ylabel('Density')
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'      Chart saved: {out_path}')


def chart_transactions_per_account(df, out_path):
    normal_df     = df[df['is_fraud'] == 0]
    fraud_df      = df[df['is_fraud'] == 1]
    normal_counts = normal_df.groupby('sender_id').size().clip(upper=200)
    fraud_counts  = fraud_df.groupby('sender_id').size().clip(upper=200)
    fig, ax = _common_fig()
    ax.hist(normal_counts, bins=60, color=GREY_NORMAL, alpha=0.75,
            label='Normal senders', density=True, edgecolor='none')
    ax.hist(fraud_counts,  bins=60, color=RED_FRAUD,   alpha=0.85,
            label='Fraud senders',  density=True, edgecolor='none')
    ax.set_title('Outgoing Transactions per Account (clipped at 200)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Outgoing Transaction Count')
    ax.set_ylabel('Density')
    ax.legend(framealpha=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'      Chart saved: {out_path}')


def generate_charts(df):
    print('\n[Charts] Generating diagnostic charts ...')
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart_fraud_vs_normal_over_time(df, DOCS_DIR / 'fraud_vs_normal_over_time.png')
    chart_amount_distribution(df,       DOCS_DIR / 'amount_distribution.png')
    chart_transactions_per_account(df,  DOCS_DIR / 'transactions_per_account.png')


def compute_split_stats(transactions, ring_info):
    # Per-window fraud stats match train_tgnn.py build_all_snapshots logic:
    # a node is labelled fraud in a window only if it transacts fraudulently there.
    txn       = transactions.copy()
    t_min     = txn['timestamp'].min()
    window_td = pd.Timedelta(minutes=WINDOW_MINUTES)
    total_windows = int(((txn['timestamp'].max() - t_min) // window_td)) + 1
    n_train = int(total_windows * TRAIN_FRAC)
    n_val   = int(total_windows * VAL_FRAC)
    n_test  = total_windows - n_train - n_val
    train_end = t_min + n_train * window_td
    val_end   = train_end + n_val * window_td

    splits = {
        'TRAIN': (t_min,     train_end),
        'VAL':   (train_end, val_end),
        'TEST':  (val_end,   txn['timestamp'].max() + pd.Timedelta(seconds=1)),
    }

    print('\n' + '=' * 70)
    print('PER-SPLIT FRAUD STATISTICS  (per-window ground truth)')
    print('=' * 70)
    print(f'Total windows: {total_windows}  (train={n_train}, val={n_val}, test={n_test})')

    ring_split_counts = {'TRAIN': 0, 'VAL': 0, 'TEST': 0}

    for split, (t_start, t_end) in splits.items():
        st = txn[(txn['timestamp'] >= t_start) & (txn['timestamp'] < t_end)]
        ft = st[st['is_fraud'] == 1]
        fa = set(ft['sender_id']) | set(ft['receiver_id'])
        pct = len(ft) / len(st) * 100 if len(st) else 0.0
        print(f'  [{split}]  {t_start.strftime("%d-%b %H:%M")} to {t_end.strftime("%d-%b %H:%M")}')
        print(f'    Total txns     : {len(st):,}')
        print(f'    Fraud txns     : {len(ft):,}  ({pct:.2f} %)')
        print(f'    Fraud accounts : {len(fa)}')
        print()

    print('  [RING PLACEMENT TABLE]')
    print('  {:>4}  {:>20}  {:>5}  {:>5}'.format('Ring', 'Burst Start', 'Split', 'Mules'))
    print('  ' + '-' * 44)
    for ri in ring_info:
        bs = ri['burst_start']
        if   bs < train_end: spl = 'TRAIN'
        elif bs < val_end:   spl = 'VAL'
        else:                spl = 'TEST'
        ring_split_counts[spl] += 1
        print('  {:>4}  {:>20}  {:>5}  {:>5}'.format(
            ri['ring_idx'], bs.strftime('%d-%b %H:%M'), spl, ri['n_mules']))

    print()
    print(f'  Ring counts per split : {ring_split_counts}')
    print('=' * 70)

    for spl, cnt in ring_split_counts.items():
        assert cnt >= 1, f'Split {spl} has 0 rings!'
    print('  [OK] All splits contain at least 1 ring.')

    t_start_test, t_end_test = splits['TEST']
    test_ft = txn[(txn['timestamp'] >= t_start_test) &
                  (txn['timestamp'] <  t_end_test) &
                  (txn['is_fraud']  == 1)]
    test_fa = set(test_ft['sender_id']) | set(test_ft['receiver_id'])
    print(f'  Test fraud accounts : {len(test_fa)}')
    if len(test_fa) >= 30:
        print('  [OK] Test split has >= 30 fraud accounts.')
    else:
        print(f'  [WARN] Only {len(test_fa)} test fraud accounts (target >= 30).')
    return ring_split_counts


def print_summary(accounts, transactions, txn_path, acct_path, ring_split_counts):
    n_accts       = len(accounts)
    n_txns        = len(transactions)
    n_fraud       = int(transactions['is_fraud'].sum())
    frac          = n_fraud / n_txns * 100
    t_min         = transactions['timestamp'].min()
    t_max         = transactions['timestamp'].max()
    n_fraud_accts = int(accounts['is_fraud_account'].sum())

    print('\n' + '=' * 70)
    print('PHASE 1 VERIFICATION SUMMARY  (v2 - INR)')
    print('=' * 70)
    print(f'  Account count       : {n_accts:,}')
    print(f'  Fraud accounts      : {n_fraud_accts:,}')
    print(f'  Transaction count   : {n_txns:,}')
    print(f'  Fraud count         : {n_fraud:,}')
    print(f'  Fraud percentage    : {frac:.2f} %')
    print(f'  Time range          : {t_min}  to  {t_max}')
    print(f'  Currency            : INR (synthetic benchmark)')
    print(f'  transactions.csv    : {txn_path}')
    print(f'  accounts.csv        : {acct_path}')
    print(f'  Charts              : {DOCS_DIR}')
    print(f'  Rings per split     : {ring_split_counts}')
    print('=' * 70)

    assert n_accts >= 1_800, f'Too few accounts: {n_accts}'
    assert n_txns  >= 40_000, f'Too few transactions: {n_txns}'
    assert 2.0 <= frac <= 5.0, f'Fraud fraction out of range: {frac:.2f} %'
    print('  All spec checks passed.')


if __name__ == '__main__':
    accounts, transactions, all_fraud_accts, ring_info = assemble_dataset()
    txn_path, acct_path = save_csvs(accounts, transactions)
    generate_charts(transactions)
    ring_split_counts = compute_split_stats(transactions, ring_info)
    print_summary(accounts, transactions, txn_path, acct_path, ring_split_counts)
