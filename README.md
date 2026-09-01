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

## 🏃 Project Structure & Running the Code

```
ml-transaction-fraud-detection/
├── src/                        # Feature Engineering Pipeline
│   └── features.py             # Leakage-safe fit/transform & rolling features
├── models/                     # Saved Model Assets & Preprocessors
│   ├── feature_state.pkl
│   ├── calibrated_xgb.pkl
│   ├── xgb_baseline.pkl
│   ├── stacking_weights.pkl
│   └── fraud_mlp.pt
├── images/                     # Generated Visualizations
│   └── walk_forward_trend.png  # Walk-forward validation temporal trend
├── fraud_detection.py          # Main End-to-End Dual-Layer Pipeline Script
├── score_batch.py              # CLI Batch Inference Script (with operational_action)
├── predict_transaction.py      # Interactive Real-Time Simulator (ALLOW/CHALLENGE/HARD_BLOCK)
├── fraud_detection.ipynb       # Project Jupyter Notebook
├── README.md                   # Dual-Layer Architecture Documentation
└── requirements.txt            # Dependencies
```

### Running the Code

```bash
# Run full pipeline with dual-layer evaluation (Statistical + Business)
python fraud_detection.py

# Batch scoring CLI (outputs fraud_probability and operational_action)
python score_batch.py --input data/new_transactions.csv --output data/scored.csv --threshold 0.07

# Interactive simulator CLI
python predict_transaction.py
```

---

## 📜 License & Acknowledgments

* Dataset provided by **IEEE Computational Intelligence Society (IEEE-CIS)** and **Vesta Corporation**.
* Built by [Anish](https://github.com/anish435).
