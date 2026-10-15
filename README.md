# PulseGuard-Z

**Temporal Graph Neural Network (TGNN) for G2P Welfare Flash Fraud Detection on IBM LinuxONE**  
*IBM Z Hackathon | Event Code: SSF2026 | Team: Error-404*

---

## 1. Overview & Problem Statement

Government-to-Person (G2P) social welfare programs transfer vital funds to millions of citizens. However, organized criminal syndicates increasingly exploit disbursement events using **automated flash fraud**:
- **Dispersal & Funneling:** Fraudulent syndicates hijack beneficiary credentials, routing micro-deposits through networks of short-lived money-mule accounts within minutes of transfer release.
- **Topological Camouflage:** Mules interleave illicit welfare funds with legitimate merchant and agent activity, defeating traditional per-account volume threshold checks.
- **Temporal Velocity:** By the time batch rule audits run hours or days later, the funds have already been layered and withdrawn at off-network endpoints.

**PulseGuard-Z** solves this challenge by modeling welfare disbursements as dynamic temporal graph snapshots evaluated by a lightweight **Temporal Graph Neural Network (TGNN)** implemented in native CPU PyTorch. Running colocated on **IBM LinuxONE** mainframes alongside transaction databases, PulseGuard-Z detects multi-hop mule forwarding rings in sub-15ms, stopping flash fraud cascades before funds exit the payment perimeter.

---

## 2. Key Implemented Features

- **Dynamic Temporal Graph Analysis:** Evaluates 144 discrete 30-minute graph snapshot windows with volume-weighted transaction Laplacian normalization ($A_{\text{norm}} = D^{-1/2} A D^{-1/2}$).
- **Lightweight Hybrid Deep Architecture:** 2 GCN layers (hidden=32) combined with 1 GRU layer (hidden=64) concatenated with instantaneous raw behavioral features to prevent temporal over-smoothing.
- **Leakage-Safe Scaling:** `StandardScaler` fitted strictly on historical training snapshots (0..85), guaranteeing that validation (86..120) and test (121..143) horizons remain completely out-of-sample.
- **Multi-Model Benchmark & Ensembling:** Comprehensive side-by-side evaluation of TGNN ($F_1=0.7174$), Logistic Regression ($F_1=0.7135$), Rule Baseline ($F_1=0.3937$), and Hybrid Ensemble ($F_1=0.7826$).
- **Multi-Signal AML Explainability:** Translates deep graph topological anomalies into human-interpretable typologies (e.g., *Rapid Funneling / Pass-Through Velocity*, *Burst Influx Ratio*, *Disproportionate Forwarding*).
- **Hardened In-Memory Operator Authentication:** PBKDF2-HMAC-SHA256 password hashing (260,000 iterations), constant-time credential verification, IP-based brute-force rate limiting (5 failures per 5 minutes), and in-memory Bearer session tokens with 30-minute TTL (zero browser storage leakage).
- **Interactive SOC Investigation Dashboard:** Single-screen Cytoscape.js network visualization, real-time threshold calibration slider ($\tau=0.05 \dots 0.99$), priority alert queues, and safe chronological simulation replay.

---

## 3. Architecture Overview

```mermaid
flowchart TD
    subgraph DataPipeline [1. Synthetic Data & Temporal Feature Pipeline]
        A[G2P Benchmark Data<br/>50k Txns / 2k Accounts] --> B[30-min Temporal Graph Snapshots<br/>144 Windows]
        B --> C[Leakage-Safe Scaler<br/>Fitted on Train 0..85]
        C --> D[Volume-Weighted Sparse Adjacency<br/>log1p amount + 1.0]
    end

    subgraph ModelEngine [2. Machine Learning Engine - Plain PyTorch CPU]
        D --> E[Spatial GCN Layers<br/>2 Layers, Hidden=32]
        E --> F[Temporal GRU Recurrence<br/>T=6 Lookback, Hidden=64]
        F --> G[Instantaneous Feature Concat<br/>H_GRU || X_raw]
        G --> H[2-Layer MLP Head & Sigmoid<br/>Topological Risk Score]
    end

    subgraph ServiceLayer [3. Operational Service Layer - FastAPI]
        H --> I[Backend Data Repository<br/>Fast Memory Lookups]
        I --> J[REST API Endpoints<br/>/dashboard, /accounts, /graph, /explain]
        K[Dual Auth Engine<br/>PBKDF2-HMAC + In-Memory Sessions] --> J
        L[IP Rate Limiter<br/>5 attempts / 5 min] --> K
    end

    subgraph InterfaceLayer [4. Analyst Dashboard - Cytoscape.js]
        J --> M[Interactive Network Topology<br/>Mule Cluster Visualization]
        J --> N[Real-Time Risk Re-ranking<br/>Threshold Slider tau]
        J --> O[Account Investigation Drawer<br/>AML Typology Explanations]
    end
```

---

## 4. Technology Stack & Deployment Philosophy

| Component | Technology | Rationale |
|:---|:---|:---|
| **Core ML Framework** | **Plain PyTorch (`torch>=2.2.0`)** | **No PyTorch Geometric (PyG) or CUDA extensions.** Eliminates complex C++ compiler requirements on IBM LinuxONE (`s390x`), running CPU inference natively with sub-15ms cold start. |
| **Feature Processing** | **NumPy, Pandas, Scikit-learn** | High-performance vectorized tabular aggregations and standard chronological windowing. |
| **REST Backend** | **FastAPI, Pydantic v2, Uvicorn** | Asynchronous ASGI architecture delivering sub-millisecond API response times and self-documenting OpenAPI schemas. |
| **Security & Auth** | **Python Standard Library** | `hashlib` (PBKDF2-HMAC-SHA256), `secrets` (constant-time comparison, token generation), and `threading` (thread-safe session cache). Zero external auth dependencies. |
| **Frontend UI** | **HTML5, Vanilla CSS, Vanilla JS, Cytoscape.js** | Single-file zero-framework architecture (`frontend/index.html`). No Node.js build pipeline or npm vulnerabilities. Tokens stored strictly in JS closure memory. |

---

## 5. Repository Structure

```
-Error-404-PulseGuard-Z/
├── .env.example             # Documented template for local and production environment variables
├── .gitignore                # Strictly excludes .env, virtualenvs, cache, and compiled models
├── README.md                 # Primary project documentation, architecture, and usage guide
├── requirements.txt          # Python runtime dependencies
├── run_all.sh                # End-to-end execution pipeline script (Linux/Unix)
│
├── backend/                  # Operational FastAPI REST Service
│   ├── auth.py               # PBKDF2 hashing, session management, IP rate limiter, dual auth
│   ├── explain.py            # Multi-signal AML explainability typologies and narrative generator
│   └── main.py               # API endpoints (/health, /auth/*, /dashboard/*, /accounts/*, /graph)
│
├── data/                     # Benchmark Data Storage (Local CSVs)
│   ├── accounts.csv          # 2,000 synthetic account profiles with account types and prior fraud labels
│   ├── transactions.csv      # 50,000 chronological transactions spanning 144 snapshot windows
│   ├── ring_membership.csv   # Ground-truth synthetic fraud mule ring definitions
│   └── scores.csv            # Pre-computed model scores (TGNN, LR, Rule, Hybrid) across all accounts
│
├── docs/                     # Research Documentation, Assets & Operational Guides
│   ├── DECISIONS.md          # Comprehensive architectural and user experience design records
│   ├── DEPLOYMENT.md         # Production deployment guide for IBM LinuxONE (s390x) and NGINX
│   ├── results.md            # Verified Phase 2 evaluation metrics, confusion matrices, and audit notes
│   ├── roc_curve.png         # Out-of-sample ROC evaluation curve chart
│   └── pr_curve.png          # Precision-Recall curve comparison chart
│
├── frontend/                 # Zero-Framework Web Dashboard
│   └── index.html            # Monochromatic/neon-green dashboard with Cytoscape.js graph and login modal
│
├── ml/                       # Machine Learning Pipeline
│   ├── generate_data.py      # Synthetic G2P welfare disbursement and mule ring simulation generator
│   ├── train_tgnn.py         # TGNN training loop with validation-only 4-config grid tuning
│   └── evaluate.py           # Multi-model evaluation pipeline and results.md generator
│
└── scripts/                  # Operations & Maintenance Utilities
    └── hash_password.py      # Standard-library PBKDF2-HMAC password hashing utility for operators
```

---

## 6. Installation & Configuration

### Prerequisites
- **Python:** Python 3.10, 3.11, or 3.12 (Python 3.9+ supported).
- **Environment:** Windows (PowerShell) or Linux / IBM LinuxONE (`s390x`).

### Windows PowerShell Setup
```powershell
# 1. Clone the repository
git clone https://github.com/lkishan2008-bit/-Error-404-PulseGuard-Z.git
cd -Error-404-PulseGuard-Z

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Linux / IBM LinuxONE Setup
```bash
git clone https://github.com/lkishan2008-bit/-Error-404-PulseGuard-Z.git
cd -Error-404-PulseGuard-Z

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Configuring Authentication & Starting the Server

PulseGuard-Z includes a secure, zero-dependency operator authentication layer.

### Step 1: Generate an Operator Password Hash
Run the included password-hash helper script:
```powershell
python scripts/hash_password.py "YourSecurePassword"
```
*Output will display a salt and PBKDF2 hash formatted as `salt_hex:hash_hex`.*

### Step 2: Create Local `.env` Configuration
Copy the template and configure your local settings:
```powershell
Copy-Item .env.example .env
```
Edit `.env` (this file is excluded from Git by `.gitignore`):
```ini
# Legacy API Key Authentication (optional for local testing)
API_KEY=local-dev-api-key-2026
REQUIRE_API_KEY=false

# Dashboard Operator Authentication
DASHBOARD_USER=admin
DASHBOARD_PASSWORD_HASH=<paste generated salt_hex:hash_hex from Step 1>
SESSION_TTL_MINUTES=30

# CORS Allowed Origins (restrict to your dashboard host)
CORS_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000

# Trusted Proxies (leave empty if testing locally without NGINX)
TRUSTED_PROXIES=
```

### Step 3: Start the FastAPI Backend Service
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
The server will initialize data repositories from `data/*.csv` and begin listening on `http://127.0.0.1:8000`.

---

## 8. Accessing the Dashboard & Authentication

1. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```
2. **Operator Login:**
   - If `DASHBOARD_USER` and `DASHBOARD_PASSWORD_HASH` are configured in `.env`, the **Operator Authentication** modal will prompt for credentials.
   - Enter your configured username (e.g., `admin`) and password.
   - Upon successful verification, the modal will dismiss, storing an unguessable 32-byte Bearer token **exclusively in JavaScript memory** (never in `localStorage` or cookies).
3. **Investigate Fraud Mules:**
   - Explore the Cytoscape transaction topology graph. Red nodes represent flagged accounts ($\ge \tau$).
   - Click any node or priority alert to inspect its multi-signal AML explainability profile and recent transactional flow.
   - Adjust the decision threshold slider ($\tau=0.81$ recommended) to observe dynamic risk recalibration.
4. **Sign Out:** Click the red **Sign Out** button in the sidebar footer to immediately revoke the session token on the server and return to the login screen.

---

## 9. Machine Learning Pipeline: Data Generation, Training & Evaluation

If you wish to re-generate synthetic benchmark data, re-train the TGNN model, or re-run evaluation:

> [!IMPORTANT]
> Always execute the pipeline commands in the exact order below from the repository root.

### Step 1: Generate Synthetic Benchmark Data
Generates 50,000 transactions across 2,000 accounts with simulated G2P welfare fraud rings:
```powershell
python ml/generate_data.py
```
*Outputs: `data/accounts.csv`, `data/transactions.csv`, and `data/ring_membership.csv`.*

### Step 2: Train Temporal Graph Neural Network (TGNN)
Executes CPU-optimized training with early stopping across a 4-configuration grid search over learning rates and positive class weights:
```powershell
python ml/train_tgnn.py
```
*Outputs: Best model checkpoint saved to `ml/tgnn_model.pt`.*

### Step 3: Evaluate Models and Export Risk Scores
Evaluates all four models on the untouched test horizon (snapshots 121–143), exports scores, and computes performance curves:
```powershell
python ml/evaluate.py
```
*Outputs: `data/scores.csv`, `docs/results.md`, `docs/roc_curve.png`, and `docs/pr_curve.png`.*

---

## 10. Summary of Verified Evaluation Results

All models were evaluated on all **2,000 accounts** during the untouched test horizon (snapshots 121–143) containing **102 ground-truth fraud accounts** (5.10% test prior). Decision thresholds were calibrated exclusively on validation data:

| Model | Calibrated $\tau^*$ | Precision | Recall | $F_1$ Score | ROC-AUC | PR-AUC | Flagged Accounts | Ring Detection Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TGNN (Ours)** | **0.81** | **0.8049** | **0.6471** | **0.7174** | **0.9673** | **0.7684** | **82** | **85.7% (12/14)** |
| **Logistic Regression** | 0.60 | 0.7952 | 0.6471 | 0.7135 | 0.9821 | 0.8136 | 83 | 85.7% (12/14) |
| **Rule Baseline** | 0.50 | 1.0000 | 0.2451 | 0.3937 | 0.6225 | 0.2836 | 25 | 42.9% (6/14) |
| **Hybrid Ensemble** | **0.63** | **0.8780** | **0.7059** | **0.7826** | **0.9893** | **0.8618** | **82** | **78.6% (11/14)** |

### Note on Validation $F_1$ Score Distinction
- **In-Training Grid Search ($F_1=0.4348$):** In `ml/train_tgnn.py`, the training grid search evaluated sequence windows exclusively inside the isolated validation partition (`val_s[0..34]`). With sequence length $T=6$, target snapshot evaluation only began at index 5 (snapshot 91 onward), omitting snapshots 86–90 where initial fraud bursts occurred.
- **Full Validation Calibration ($F_1=0.9273$ at $\tau^*=0.81$):** In `ml/evaluate.py`, `tgnn_inference()` applies continuous look-back across the train-val boundary ($t_{\text{abs}}=86$ evaluates using snapshots 81..86), evaluating all 35 validation snapshots (86..120) and all 108 validation fraud accounts to identify $\tau^*=0.81$.
- *Both figures reflect genuine computations from their respective code paths and are documented in [`docs/results.md`](file:///c:/Users/hp/Documents/projects/-Error-404-PulseGuard-Z/docs/results.md).*

---

## 11. Security Hardening Architecture

- **Header-Only Authentication:** Following security audit hardening in Step 5A, credentials are accepted strictly via headers (`Authorization: Bearer <token_or_key>` and `X-API-Key: <key>`). The insecure `?api_key=` URL query parameter has been disabled to prevent credential leakage in web server access logs and browser histories.
- **Constant-Time Verification:** All password and API key checks utilize `secrets.compare_digest()`, mitigating timing side-channel attacks.
- **Brute-Force Rate Limiting:** An in-memory IP rate limiter tracks failed login attempts. Any client IP triggering 5 consecutive failed logins within a 5-minute rolling window receives `HTTP 429 Too Many Requests`.
- **Anti-Spoofing Proxy Validation:** `get_client_ip()` checks `X-Forwarded-For` **only** if the direct peer IP is explicitly declared in `TRUSTED_PROXIES`. Untrusted external clients cannot bypass rate limiting by spoofing forward headers.
- **In-Memory Session Security:** Session tokens are stored strictly in Python and JavaScript application memory. Tokens are never written to `localStorage`, `sessionStorage`, cookies, or version-controlled files.
- **Explicit CORS Allowlist:** Wildcard `*` origins are eliminated in favor of an environment-configurable allowlist (`CORS_ALLOWED_ORIGINS`).

---

## 12. Honest Limitations & Operational Scope

1. **Synthetic Simulation Benchmark:** All transactions, accounts, and fraud rings are synthetically generated to model G2P social welfare disbursement patterns under IBM LinuxONE constraints (event code SSF2026). They do not represent live citizen records.
2. **In-Memory Session Volatility:** Active operator login sessions are held in server memory. Restarting the Uvicorn process invalidates active tokens. For multi-worker production deployments, an external session store (e.g., Redis on Linux on Z) is recommended.
3. **Sidebar View Scope:** To maintain complete institutional honesty, unfinished navigation sections (`Payment Demo`, `Cyber Risk SOC`) are flagged with `PLANNED` badges. The `AI Reports` item links directly to [`docs/results.md`](file:///c:/Users/hp/Documents/projects/-Error-404-PulseGuard-Z/docs/results.md), and `Transactions` reflects the active backend API (`/api/transactions`).

---

## 13. Production Deployment on IBM LinuxONE

For production deployment instructions on **IBM LinuxONE (`s390x`)**, including RHEL systemd service configuration, CPU-only PyTorch setup, and NGINX reverse-proxy TLS termination, please consult the complete deployment guide:

👉 [**docs/DEPLOYMENT.md**](docs/DEPLOYMENT.md)

---

## 14. License & Event Acknowledgements

Developed by **Team Error-404** for the **IBM Z Hackathon (SSF2026)**.
Licensed under the Apache License 2.0.
