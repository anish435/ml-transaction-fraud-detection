# Credit Card Fraud Detection Engine — Dual-Layer Evaluation Architecture

An end-to-end, enterprise-grade machine learning system designed to detect credit card transaction fraud on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) benchmark (~590K transactions, 27.6:1 class imbalance). 

This repository evaluates fraud detection through two distinct, complementary layers:
1. **Statistical Model Performance Layer**: PR-AUC, ROC-AUC, Isotonic Brier score probability calibration, and walk-forward validation.
2. **Business Decision Layer**: Flat $5 penalty cost sweep vs. Dynamic value-based financial loss, Three-Tiered Operational Action Routing (`ALLOW`, `CHALLENGE`, `HARD_BLOCK`), and human-readable SHAP alerts.

---

## 📊 SECTION 1: STATISTICAL MODEL PERFORMANCE LAYER

### 1. Sealed Test Set Metrics (Algorithmic Discrimination)

| Model / Ensemble Blend | Sealed Test ROC-AUC | Sealed Test PR-AUC | Model Family Rank |
| :--- | :---: | :---: | :---: |
| Logistic Regression Baseline | 0.8272 | 0.1761 | 4 |
| PyTorch MLP Neural Network | 0.8271 | 0.1922 | 3 |
| Baseline XGBoost Classifier (No Velocity) | 0.8900 | 0.5026 | 2 |
| Calibrated XGBoost + Velocity + Diversity | **0.8929** | **0.5100** | 1 (Single) |
| **Stacked Ensemble Blend (Final)** | **0.8884** | **0.5116** | **1 (Overall)** |

### 2. Probability Calibration Quality (Validation Set)

| Method | Brier Loss *(Lower = Better)* | Precision @ 0.30 | Recall @ 0.30 | False Alarms @ 0.30 | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Uncalibrated XGBoost** | 0.068670 | 13.97% | 83.66% | 15,672 | 0.5399 |
| **Isotonic Calibrated XGBoost** *(Selected)* | **0.021501** (-68.7%) | **67.38%** | 43.20% | **636** (-96.2%) | 0.5313 |
| **Sigmoid Calibrated XGBoost** | 0.022425 | 54.33% | 49.67% | 1,270 | 0.5399 |

---

## 💼 SECTION 2: BUSINESS DECISION & FINANCIAL RISK LAYER

### 1. Financial Cost Models Comparison

#### A. Flat $5 Fixed False-Alarm Cost Model
$$\text{Cost} = \sum_{\text{Missed Fraud (FN)}} \text{TransactionAmt} + \text{False Alarms (FP)} \times \$5$$
* **Optimal Threshold**: **`0.03`**
* **Total Flat Financial Loss**: **`$136,324.12`**
* **Precision**: `13.79%` | **Recall**: `83.86%` (`2,551` TP / `15,952` FP)

#### B. Dynamic Value-Based Financial Loss Model
$$\text{FP Cost} = (\text{TransactionAmt} \times 0.03) + \$10.00 \text{ (Churn Risk Penalty)}$$
$$\text{FN Cost} = \text{TransactionAmt} + \$15.00 \text{ (Chargeback Fee)}$$
* **Optimal Dynamic Threshold**: **`0.07`**
* **Total Dynamic Financial Loss**: **`$242,451.69`** (vs $296,728.41 @ 0.03 threshold)
* **Precision**: **`29.75%`** (more than double the binary 14.63% precision!)
* **Recall**: **`66.70%`** (`2,029` TP / `4,791` FP — **67.4% false-alarm reduction**)

---

### 2. Three-Tiered Operational Action Routing Evaluation

Evaluated across validation set (`88,581` transactions, `$11,947,493.75` total GMV):

| Risk Zone | Probability Range | Operational Action | Volume % | Transactions | GMV Approved / Processed ($) | Operational Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW RISK** | `p < 0.03` | **`ALLOW`** | **80.58%** | 71,378 | **$8,826,759.00** | Frictionless path (**73.9% of GMV**) |
| **MEDIUM RISK** | `0.03 <= p < 0.30` | **`CHALLENGE`** | **17.22%** | 15,253 | $2,838,225.50 | 2FA / Step-Up OTP (recovers honest GMV) |
| **HIGH RISK** | `p >= 0.30` | **`HARD_BLOCK`** | **2.20%** | 1,950 | $282,509.25 | Rejects extreme fraud (**67.38% Precision**) |

* **Hard Block Precision (High Risk Zone)**: **`67.38%`** (2 in 3 blocked transactions are confirmed fraud).
* **Total Fraud Identified (Med + High Zones)**: **`82.74%`** of all fraudulent volume.
* **Frictionless Approved GMV (Low Risk Zone)**: **`$8,826,759.00`** approved without friction.

---

## 🛠️ System Architecture & Workflow

```
                   Raw IEEE-CIS Data (590k Rows)
                                 │
                 Chronological Sorting (TransactionDT)
                                 │
     Pre-Split Past-Only Velocity & Counterparty Diversity Features
    (card_tx_count_1h, card_tx_count_24h, card_counterparty_diversity_24h)
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      ▼                          ▼                          ▼
Train Set (70%)           Val Set (15%)              Test Set (15%)
(Rows 0 - 413,378)     (Rows 413,378 - 501,959)   (Rows 501,959 - 590,540)
      │                          │                          │
  Train Stats                Fit Calibration            Sealed Final
 (fit_feature_pipeline)   & Stack Weights             Evaluation
      │                          │                          │
      └──────────────────────────┴──────────────────────────┘
                                 │
      Model Suite: Logistic Regression | PyTorch MLP | XGBoost
                                 │
         Isotonic Calibration ──► Ensemble Stacking (Blend)
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
   SECTION 1: ML PERFORMANCE             SECTION 2: BUSINESS DECISIONS
(PR-AUC: 0.5116, ROC-AUC: 0.8929)     (3-Tiered Routing & Dynamic Cost)
```

---

## 🔍 SHAP Explainability & Operational Alerts

| Feature Name | Category | SHAP Rank | Description |
| :--- | :--- | :---: | :--- |
| `C13`, `C14`, `C5`, `C1` | Vesta Counters | **Ranks 1–5** | Main transaction count indicators |
| `TransactionAmt` | Financial | **Rank 3** | Transaction dollar amount |
| `card_amt_mean` | Engineered | **Rank 7** | Card historical average amount |
| `card_amt_std` | Engineered | **Rank 15** | Card historical amount standard deviation |
| `card_tx_count_24h` | Velocity | **Rank 36** | 24-hour transaction count |
| `amt_z_for_card` | Engineered | **Rank 37** | Dollar amount z-score for card |
| `card_counterparty_diversity_24h` | Diversity | **Rank 43** | Ratio of distinct domains to 24h count |
| `card_time_since_last_tx` | Velocity | **Rank 49** | Seconds elapsed since previous transaction |

### Operational Risk Alert Helper (`explain_transaction_alert`)
* `[!] Extreme 24-Hour Velocity (5 tx in last 24 hours)`
* `[!] High Email Domain Diversity (2.50 domain diversity ratio)`
* `[!] Rapid Transaction Repeat (only 12.4s since last tx)`
* `[!] Unusual Amount for Card Profile (3.45 z-score relative to card mean)`

---

## 📉 Population Stability Index (PSI) Monitoring

Monitors distribution drift between baseline training data and future time-based evaluation splits/windows using decile binning:
$$\text{PSI} = \sum_{i=1}^{k} \left(\text{Actual}\%_i - \text{Expected}\%_i\right) \times \ln\left(\frac{\text{Actual}\%_i}{\text{Expected}\%_i}\right)$$

| Target Window / Split | Calibrated Score | C13 | C14 | TransactionAmt | C5 | C1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set (70–85%)** | **0.0014** | 0.0366 | 0.0125 | 0.0087 | 0.0172 | 0.0095 | **STABLE** |
| **Sealed Test Set (85–100%)** | **0.0015** | 0.0259 | 0.0073 | 0.0048 | 0.0029 | 0.0120 | **STABLE** |
| **WF Window 1 (10–25%)** | **0.0037** | 0.0543 | 0.0268 | 0.0273 | 0.0388 | 0.0259 | **STABLE** |
| **WF Window 2 (25–40%)** | **0.0055** | 0.0124 | 0.0053 | 0.0106 | 0.0048 | 0.0046 | **STABLE** |
| **WF Window 3 (40–55%)** | **0.0024** | 0.0101 | 0.0036 | 0.0062 | 0.0132 | 0.0042 | **STABLE** |
| **WF Window 4 (55–70%)** | **0.0025** | 0.0074 | 0.0030 | 0.0073 | 0.0042 | 0.0026 | **STABLE** |

*Thresholds*: $\text{PSI} < 0.10$ (**Stable**), $0.10 \le \text{PSI} < 0.25$ (**Moderate**), $\text{PSI} \ge 0.25$ (**[!] Significant Drift / Retrain Alert**).

---

## 🚀 Production Product Layer (FastAPI + Streamlit + Render)

The system includes a production-grade inference microservice and an analyst dashboard for real-time fraud intervention:

### 1. FastAPI Real-Time Microservice (`src/api/main.py`)
- **Endpoints**:
  - `POST /score`: Accepts single transaction payload, executes inference, and outputs calibrated probability, 3-tier routing action, top 3 logically-accurate SHAP alerts, and latency in ms (optional `?include_reasons=false`).
  - `POST /score-fast` & `GET /score-fast`: Ultra-fast inference bypassing SHAP (~120ms single-row feature expansion vs ~198ms with SHAP).
  - `GET /health`: Uptime and model readiness status.
  - `GET /stats`: In-memory comparative rolling latency tracking ($p_{50}$, $p_{95}$, $p_{99}$) for both fast mode and explainable mode over the last 100 requests.
- **Port Support**: Automatically binds to `$PORT` (default: 8000) for Render compatibility.

```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Streamlit Risk Analyst Dashboard (`dashboard/app.py`)
- **Features**:
  - **1-Click Judge Presets**:
    - 🟢 `Scenario 1: Low Risk (Grocery)` $\rightarrow$ **`ALLOW`** ($p = 0.0038$)
    - 🟡 `Scenario 2: Elevated Risk Spike` $\rightarrow$ **`CHALLENGE (2FA / OTP)`** ($p = 0.1911$)
    - 🔴 `Scenario 3: Extreme Syndicate Attack` $\rightarrow$ **`HARD_BLOCK`** ($p = 0.7778$)
  - Real-time probability progress gauge & color-coded risk action badges.
  - Actionable SHAP operational alerts bullet list.
  - Live comparative latency SLA telemetry.

```bash
# Start Streamlit Dashboard (connects to FastAPI via API_BASE_URL)
streamlit run dashboard/app.py
```

---

## 🏃 Project Structure & Running the Code

```
ml-transaction-fraud-detection/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI Microservice (POST /score, GET /health, GET /stats)
│   ├── features.py             # Leakage-safe fit/transform & rolling features
│   └── monitoring.py           # Standalone Population Stability Index (PSI) engine
├── dashboard/
│   ├── app.py                  # Streamlit Risk Analyst Dashboard
│   └── presets.json            # 1-Click verified judge demo presets
├── models/                     # Saved Model Assets & Preprocessors
│   ├── feature_state.pkl
│   ├── calibrated_xgb.pkl
│   ├── xgb_baseline.pkl
│   ├── stacking_weights.pkl
│   ├── train_score_sample.npy
│   └── fraud_mlp.pt
├── images/                     # Generated Visualizations
│   └── walk_forward_trend.png  # Walk-forward validation temporal trend
├── render.yaml                 # Render web service infrastructure-as-code
├── fraud_detection.py          # Main End-to-End Dual-Layer Pipeline Script
├── score_batch.py              # CLI Batch Inference Script (supports --check-psi)
├── predict_transaction.py      # Interactive Real-Time Simulator (ALLOW/CHALLENGE/HARD_BLOCK)
├── fraud_detection.ipynb       # Project Jupyter Notebook
├── README.md                   # Dual-Layer Architecture Documentation
└── requirements.txt            # Dependencies
```

### Running the Code

```bash
# 1. Run full pipeline with dual-layer evaluation (Statistical + Business)
python fraud_detection.py

# 2. Launch FastAPI Inference Microservice
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 3. Launch Streamlit Risk Dashboard
streamlit run dashboard/app.py

# 4. Batch scoring CLI (outputs fraud_probability and operational_action)
python score_batch.py --input data/new_transactions.csv --output data/scored.csv --threshold 0.07 --check-psi

# 5. Run Razorpay End-to-End Demo Simulation
python scripts/demo_razorpay_flow.py
```

---

## 💳 SECTION 4: RAZORPAY TEST MODE INTEGRATION

This project natively integrates with **Razorpay Test Mode** to score real payment gateway transactions in real-time **without altering the trained XGBoost model or feature engineering pipeline**.

### 1. Feature Mismatch Handling Strategy

Razorpay payment objects contain gateway-level fields (`amount`, `email`, `contact`, `card`, `created_at`), while the trained ML model requires 431 IEEE-CIS features. We bridge this gap without fabricating fake data:

* **Real Overlaps Derived Live:**
  * `TransactionAmt`: Converted from paise/cents to decimal currency (`amount / 100.0`).
  * `hour`: Extracted from payment `created_at` timestamp in local time.
  * `P_emaildomain`: Extracted from customer email (e.g. `user@gmail.com` $\rightarrow$ `gmail.com`).
  * `card4` / `card6`: Extracted from `card.network` and `card.type` (`visa`, `mastercard`, `credit`, `debit`).
* **Live Customer Velocity Engine:**
  * A thread-safe, persistent customer velocity store (`data/customer_velocity_store.json`) tracks rolling timestamps per customer to dynamically compute:
    * `card_time_since_last_tx` (seconds since previous transaction)
    * `card_tx_count_1h` (1-hour attempt frequency)
    * `card_tx_count_24h` (24-hour attempt frequency)
* **Defaulted / Unseen Features:**
  * Unprovided fields (`card1`, `addr1/2`, `C1-C14`, `D1-D15`, `V1-V339`) are filled using the exact cold-start fallbacks and reindex defaults already established in `transform_features()`. **No synthetic values are fabricated.**
* **Audit Trail:**
  * Every scored payment is immutably logged to `data/razorpay_audit_log.jsonl` with an explicit manifest of real vs defaulted features, calibrated score, operational decision, and SHAP explanation alerts.

### 2. Setup & Configuration

1. **Configure Environment (`.env`):**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Populate your Razorpay Test API keys from [Razorpay Dashboard](https://dashboard.razorpay.com/#/app/keys):
   ```ini
   RAZORPAY_KEY_ID=rzp_test_YourTestKeyIdHere
   RAZORPAY_KEY_SECRET=YourTestKeySecretHere
   RAZORPAY_WEBHOOK_SECRET=your_test_webhook_secret_here
   PORT=8000
   ```

2. **Start the FastAPI Microservice:**
   ```bash
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```

3. **Start the Streamlit Risk Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

### 3. End-to-End Demo Script

Run the automated simulation script to create a test order, simulate payments, verify HMAC webhook signatures, and inspect fraud scoring:

```bash
python scripts/demo_razorpay_flow.py
```

**What the demo executes:**
1. **Creates Order:** Calls `POST /create-order` via the Razorpay Python SDK.
2. **Normal Customer Payment:** Simulates legitimate payment capture, signs webhook with HMAC SHA256, and verifies an instant **`ALLOW`** decision.
3. **Rapid Velocity Bot Attack:** Simulates 4 rapid-fire micro-charges and sudden amount spikes for the same customer card within seconds, observing velocity counters climbing and risk escalating dynamically.
4. **Security Check:** Sends a tampered webhook signature and verifies strict HTTP `400 Bad Request` rejection.
5. **Audit Inspection:** Displays the latest entries logged to `data/razorpay_audit_log.jsonl`.

### 4. Live External Webhook Setup (ngrok)

To receive live webhooks directly from Razorpay's cloud in Test Mode:

1. Expose port 8000 via ngrok:
   ```bash
   ngrok http 8000
   ```
2. In the **Razorpay Dashboard $\rightarrow$ Settings $\rightarrow$ Webhooks $\rightarrow$ Add Webhook**:
   * **Webhook URL:** `https://<your-ngrok-subdomain>.ngrok-free.app/webhook/razorpay`
   * **Secret:** Same value as `RAZORPAY_WEBHOOK_SECRET` in your `.env`.
   * **Active Events:** Check `payment.captured` and `payment.authorized`.
3. Complete a payment in Razorpay Test Mode checkout $\rightarrow$ Observe the payment scored live in the **Streamlit Dashboard $\rightarrow$ 💳 Live Razorpay Webhook Monitor** tab!


---

## 📜 License & Acknowledgments

* Dataset provided by **IEEE Computational Intelligence Society (IEEE-CIS)** and **Vesta Corporation**.
* Built by [Anish](https://github.com/anish435).
