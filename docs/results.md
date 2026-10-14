# PulseGuard-Z — Phase 2 Evaluation Results (v2 Dataset)
**IBM Z Hackathon | SSF2026 | Team: Error-404**

---

## Data Summary

| Metric | Value |
|--------|-------|
| Total accounts | 2,000 |
| Total transactions | 50,000 |
| Fraud transactions | 1,800 (3.60%) |
| Simulation period | 2026-10-01 00:00:28 → 2026-10-03 23:59:53 |
| Train-period fraud accounts | 319 |
| Val-period fraud accounts | 20 |
| Test-period fraud accounts | 102 |

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
| Weighted BCE pos_weight | ~21.9 |
| Best validation loss | 0.0814 |
| LR regularisation C | 0.1 |

---

## Threshold Selection (Validation Set)

Thresholds chosen by maximising F1 on validation data. Test set evaluated once.

| Model | Val threshold (τ*) | Val F1 |
|-------|--------------------|--------|
| TGNN | 0.15 | 0.0342 |
| Logistic Regression | 0.96 | 1.0000 |
| Rule Baseline | 0.50 (binary) | — |

---

## Test-Set Metrics

> Positive test examples: **102** / 2000 (5.10%)

| Model | τ | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy |
|-------|---|-----------|--------|----|---------|--------|----------|
| **TGNN** | 0.15 | 0.0622 | 0.2255 | 0.0975 | **0.5260** | 0.0726 | 0.7870 |
| **Logistic Regression** | 0.96 | 1.0000 | 0.4608 | 0.6309 | 0.9874 | 0.8936 | 0.9725 |
| Rule Baseline | 0.50 | 1.0000 | 0.2353 | 0.3810 | 0.6176 | 0.2743 | 0.9610 |

---

## Confusion Matrices

### TGNN (τ=0.15)
```
             Predicted 0  Predicted 1
Actual 0           1551          347
Actual 1             79           23
```

### Logistic Regression (τ=0.96)
```
             Predicted 0  Predicted 1
Actual 0           1898            0
Actual 1             55           47
```

### Rule Baseline (τ=0.50)
```
             Predicted 0  Predicted 1
Actual 0           1898            0
Actual 1             78           24
```

---

## False Positives by Account Type

| Account Type | TGNN FP | LR FP | Rule FP |
|---|---|---|---|
| agent | 27 | 0 | 0 |
| beneficiary | 281 | 0 | 0 |
| government | 6 | 0 | 0 |
| merchant | 33 | 0 | 0 |

---

## Fraud Ring Detection Rate

> **Detection definition:** A ring is *active in test* if any of its mule accounts
> appears in a fraud transaction during the test period. A ring is *detected* if
> at least one of its mule accounts is flagged (predicted positive) by the model.

| Model | Rings Active (test) | Rings Detected | Detection Rate |
|-------|---------------------|----------------|----------------|
| TGNN | 14 | 11 | 0.786 |
| Logistic Regression | 14 | 9 | 0.643 |
| Rule Baseline | 14 | 6 | 0.429 |

---

## ROC Curve
![ROC Curve](roc_curve.png)

## Precision-Recall Curve
![PR Curve](pr_curve.png)

---

## Top Signals (Highest TGNN Risk Scores)

| Account | TGNN Score | True Label | Signal 1 | Signal 2 | Signal 3 |
|---------|------------|------------|----------|----------|----------|
| ACC000460 | 0.2485 | 1 | amount_std | log_out_amount | in_count |
| ACC001989 | 0.2272 | 1 | amount_std | log_in_amount | log_out_amount |
| ACC000905 | 0.2267 | 0 | amount_std | forwarding_ratio | log_out_amount |
| ACC000570 | 0.2236 | 0 | log_in_amount | unique_senders | in_count |
| ACC000438 | 0.2216 | 0 | log_in_amount | unique_senders | in_count |
| ACC000560 | 0.2197 | 0 | log_in_amount | unique_senders | in_count |
| ACC000576 | 0.2111 | 0 | amount_std | log_out_amount | log_in_amount |
| ACC000686 | 0.2111 | 0 | forwarding_ratio | log_out_amount | out_count |
| ACC000673 | 0.2111 | 0 | forwarding_ratio | log_out_amount | out_count |
| ACC000558 | 0.2111 | 0 | amount_std | log_in_amount | in_count |
| ACC000669 | 0.2111 | 0 | forwarding_ratio | log_out_amount | out_count |
| ACC000733 | 0.2111 | 0 | amount_std | log_out_amount | forwarding_ratio |
| ACC000091 | 0.2111 | 1 | forwarding_ratio | log_out_amount | out_count |
| ACC001773 | 0.2111 | 0 | forwarding_ratio | log_out_amount | out_count |
| ACC000864 | 0.2111 | 0 | log_in_amount | unique_senders | in_count |

---

## Limitations & Honest Assessment

> [!WARNING]
> The following limitations must be understood before deploying this model.

1. **Synthetic data.** Fraud patterns were planted deliberately. Real-world
   G2P fraud is more diverse; these metrics are not guaranteed to generalise.

2. **Test class imbalance.** The test period contains 102 fraud-active
   accounts out of 2000. Precision is sensitive to false-positive count.

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
