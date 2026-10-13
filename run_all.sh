#!/usr/bin/env bash
# ==============================================================================
# PulseGuard-Z  -  Full Execution Pipeline
# Event Code   : SSF2026 (IBM Z Hackathon)
# Team         : Error-404
# ==============================================================================
set -e

echo "=========================================================="
echo " PulseGuard-Z: Flash Fraud Detection on IBM LinuxONE"
echo "=========================================================="

# Phase 1: Synthetic Data Generation
if [ ! -f "data/transactions.csv" ] || [ ! -f "data/accounts.csv" ]; then
    echo "[Phase 1] Generating synthetic transaction and account data..."
    python ml/generate_data.py
else
    echo "[Phase 1] Data files found: data/transactions.csv, data/accounts.csv"
fi

# Phase 2: TGNN Model Training
if [ ! -f "ml/tgnn_model.pt" ]; then
    echo "[Phase 2] Training Temporal Graph Neural Network (TGNN)..."
    python ml/train_tgnn.py
else
    echo "[Phase 2] Model checkpoint found: ml/tgnn_model.pt"
fi

# Phase 2: Model Evaluation & Scoring
if [ ! -f "data/scores.csv" ]; then
    echo "[Phase 2] Evaluating model and exporting fraud risk scores..."
    python ml/evaluate.py
else
    echo "[Phase 2] Scores file found: data/scores.csv"
fi

# Phase 3: Start FastAPI Backend Service
echo "[Phase 3] Starting PulseGuard-Z FastAPI Backend on http://127.0.0.1:8000 ..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
