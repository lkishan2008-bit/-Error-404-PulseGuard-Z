# PulseGuard-Z — Architectural & Design Decisions
**Event Code:** SSF2026 (IBM Z Hackathon)  
**Team:** Error-404  
**Project:** PulseGuard-Z — TGNN-Based G2P Welfare Flash Fraud Detection on IBM LinuxONE  
**File:** `docs/DECISIONS.md`

---

## 1. Executive Summary

PulseGuard-Z is an end-to-end fraud detection platform designed to detect Government-to-Person (G2P) welfare flash fraud, rapid mule rings, and automated fund drains on IBM LinuxONE enterprise infrastructure. This document outlines the architectural, technological, visual, and operational decisions governing the project across Phases 1 through 4.

---

## 2. Technology Stack Decisions

### 2.1 Plain PyTorch without PyTorch Geometric (PyG)
- **Decision:** Implement graph convolution directly in standard PyTorch using sparse COO adjacency matrices (`torch.sparse_coo_tensor`, `torch.sparse.mm`) and native PyTorch GRU layers (`nn.GRU`), entirely avoiding external graph libraries such as PyG (PyTorch Geometric) or DGL.
- **Rationale:** 
  1. **Mainframe Portability (IBM LinuxONE / s390x):** Enterprise mainframe distributions frequently lack pre-compiled C++/CUDA wheels for specialized extensions like `torch-scatter` or `torch-sparse`. Standard PyTorch CPU wheels run cleanly on Linux on IBM Z without complex compiler toolchains.
  2. **Minimal Footprint & Fast Cold-Start:** GCN (hidden=32) and GRU (hidden=64) execute in under 15 milliseconds on modern CPU cores, enabling inline inference directly adjacent to core banking transaction pipelines.
  3. **Zero Deprecation Risk:** Avoids external framework breaking changes across PyTorch version bumps.

### 2.2 FastAPI Backend Architecture
- **Decision:** Use FastAPI with Pydantic v2 and Python 3.10+ for the inference and operational REST API.
- **Rationale:**
  1. Sub-millisecond response times for transactional lookups and graph topology extractions.
  2. Native asynchronous ASGI concurrency capable of sustaining high read volumes.
  3. Self-documenting OpenAPI schemas (`/docs`) enabling instant validation for audit and compliance teams.
  4. Project-root-relative path resolution (`REPO_ROOT = Path(__file__).resolve().parent.parent`), ensuring reliability across diverse deployment environments.

### 2.3 Single-File Zero-Framework Frontend (`frontend/index.html`)
- **Decision:** Build the complete client dashboard in a single `frontend/index.html` file using Cytoscape.js via CDN, embedding all CSS in `<style>` and JavaScript in `<script>`.
- **Rationale:**
  1. **Zero Build System & Zero Dependency Drift:** No Node.js build pipeline, Webpack, Vite, or npm vulnerabilities. The file can be opened directly in any browser (`file://`) or served directly by FastAPI (`GET /`).
  2. **Instant Rendering:** Immediate DOM readiness without framework hydration delays.
  3. **Auditability:** Complete frontend logic is inspectable in a single file by security auditors.

---

## 3. User Interface & Presentation Decisions

### 3.1 Graph-First Single-Screen Layout
- **Decision:** Make the interactive transaction network graph the central visual hero (~70% screen width), paired with a compact, focused alert queue (~30% screen width) and three top-level metric cards. Eliminate sidebars, multiple pages, decorative donut charts, and progress bars.
- **Rationale:**
  1. **Immediate Comprehension for Judges & Analysts:** Fraud investigators operate visually. A graph-first layout allows an analyst or competition judge to perceive mule fan-in hubs, transit accounts, and cash-out points within three seconds of opening the dashboard.
  2. **Zero Context Switching:** Selecting a high-priority alert immediately centers and highlights the corresponding node and edges in the graph, with detailed account intelligence appearing in a smooth slide-in drawer.

### 3.2 Restrained Monochromatic Palette with Red-Only Fraud Accent
- **Decision:** Use only White (`#FFFFFF`), Light Greys (`#F9FAFB`, `#F3F4F6`, `#E5E7EB`), Charcoal (`#1F2937`, `#111827`), and a single Red accent (`#D92D20`). Strictly reject blues, greens, purples, oranges, rainbow color schemes, and decorative CSS gradients.
- **Rationale:**
  1. **Cognitive Ergonomics & Institutional Credibility:** Financial crime monitoring in mainframe enterprise environments requires extreme visual restraint (similar to Linear or Stripe).
  2. **Single Meaning for Red:** Red is reserved exclusively for risk: flagged accounts, suspicious transaction edges, and critical compliance alerts. Normal nodes and benign transactions remain neutral grey.

### 3.3 Slide-In Investigation Detail Panel
- **Decision:** Clicking any node in the Cytoscape graph or any item in the alert list slides open a dedicated investigation drawer from the right, providing:
  - Account identifier, account type, and synthetic label status.
  - TGNN risk score alongside the calibrated operational cutoff (0.20).
  - Multi-signal interpretability mapping (e.g. `log_in_amount`, `unique_senders`, `in_count`) with plain-English fraud typologies and recommended AML actions.
  - Cumulative financial flow summary (received, sent, net flow, fraud transaction count).
  - Chronological activity timeline of real transaction records.

---

## 4. Data Fidelity & Operational Metrics

### 4.1 Strict Derivation from Actual Datasets
- **Decision:** All three dashboard metric cards are computed strictly from the underlying CSV datasets (`transactions.csv`, `scores.csv`, `accounts.csv`) via FastAPI endpoints:
  - **Transactions Analysed:** `50,000` (total transactions in the 72-hour monitoring window).
  - **Accounts Flagged:** `59` (accounts evaluated with TGNN risk score ≥ 0.20 calibrated cutoff).
  - **Money at Risk:** `$3,029,457` (total transaction volume involved in flagged fraudulent flows; 17.26% of overall transaction volume).
- **Rule:** Never invent arbitrary numbers or claim synthetic data represents live, real-world government telemetry.

### 4.2 Calibrated Operating Threshold (0.20 vs Standard 0.50)
- **Decision:** Set the documented default decision threshold to `0.20` rather than standard `0.50`.
- **Rationale:** As proven in Phase 2 evaluation (`docs/results.md`), the simulation test partition exhibits extreme class imbalance (only 7 fraud accounts out of 2,000, or 0.35%). Because uncalibrated sigmoid outputs peak around ~0.2885, applying a standard 0.50 cutoff would yield zero alerts. A calibrated threshold of 0.20 surfaces 59 high-recall candidates for operational queue review.

### 4.3 Safe Replay Simulation
- **Decision:** The "Start simulation" button performs a safe, client-side sequential demo replay of existing benchmark transactions in chronological order.
- **Safety Guarantee:** It does not modify source CSVs, trigger retraining, generate fabricated records, or simulate fake API ingest.

---

## 5. IBM LinuxONE Deployment Target

PulseGuard-Z is specifically engineered for deployment on **IBM LinuxONE** mainframes:
- **Colocation with Core Banking:** LinuxONE runs enterprise Linux instances on the same physical frame as z/OS transaction processing systems. Running lightweight TGNN inference in Linux on Z eliminates cross-network latency and egress security risks.
- **Integrated High-Throughput I/O:** Fast sequential streaming of 50,000+ transaction snapshots across memory without specialized GPU accelerators.
- **Enterprise Grade Security:** API key authentication with constant-time verification (`secrets.compare_digest`) and audit-ready JSON logs.

---

*Document finalized for PulseGuard-Z Phase 4 | IBM Z Hackathon (SSF2026)*
