# Credit Card Fraud Detection Pipeline — Production Payment Gateway Engine

An end-to-end, enterprise-grade machine learning system designed to detect credit card transaction fraud on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) benchmark (~590K transactions, 27.6:1 class imbalance). 

This engine features **leakage-safe pre-split rolling velocity**, **email domain counterparty diversity**, **isotonic probability calibration**, **ensemble model stacking**, **walk-forward validation**, a **dynamic value-based financial cost model**, **three-tiered operational risk routing (`ALLOW`, `CHALLENGE`, `HARD_BLOCK`)**, and **human-readable SHAP operational alerts**.

---

## 🚀 Key Highlights & Headline Metrics

* **Sealed Test PR-AUC**: **`0.5116`** (Stacked Blend) / **`0.5100`** (Calibrated XGBoost) vs ~0.035 no-skill baseline.
* **Sealed Test ROC-AUC**: **`0.8929`** (XGBoost + Velocity + Counterparty Diversity features).
* **Probability Calibration**: Isotonic calibration reduced Brier Loss from `0.0687` to **`0.0215`** (**-68.7% loss reduction**), cutting false alarms @ 0.30 threshold by **96.2%**.
* **Dynamic Cost-Minimizing Threshold**: **`0.07`** (balancing transaction amount $ loss, 3% merchant margin, $10 churn penalty, and $15 chargeback fee), reducing total financial loss to **`$242,451.69`** and boosting Precision to **`29.75%`** (a **67.4% reduction in false alarms**).
* **Three-Tiered Operational Routing**:
  * **`ALLOW` (`p < 0.03`)**: Approves **80.58%** of total transactions (**$8.82M GMV / 73.9% of total volume**) with **zero user friction**.
  * **`CHALLENGE / STEP-UP 2FA` (`0.03 <= p < 0.30`)**: Routes **17.22%** of volume to 2FA / OTP verification, recovering honest user GMV while catching **39.5%** of fraud.
  * **`HARD_BLOCK` (`p >= 0.30`)**: Instantly rejects high-risk transactions with **`67.38%` Precision**.

---

## 📊 Dataset Overview

The dataset merges transaction logs with identity/device metadata on `TransactionID`:

* **Total Transactions**: `590,540` rows (182-day time span)
* **Class Imbalance**: `569,877` Legitimate (96.50%) vs `20,663` Fraudulent (3.50%) $\rightarrow$ **`27.6 : 1`** ratio
* **Feature Schema**:
  * `TransactionDT`: Timedelta in seconds (used for chronological sorting & rolling time-window features).
  * `TransactionAmt`: Transaction amount in USD ($).
  * `card1`–`card6`: Card identity, brand, type (debit/credit), category.
  * `addr1`, `addr2`: Billing region/zip codes.
  * `P_emaildomain`, `R_emaildomain`: Purchaser and recipient email domains.
  * `C1`–`C14`: Transaction counting features.
  * `D1`–`D15`: Timedeltas between past activity.
  * `V1`–`V339`: Vesta engineered interaction features.
  * `id_01`–`id_38`, `id_31`: Device, browser, and network identity metadata.

---

## 🛠️ Operational Architecture & Routing Strategy

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
             Dynamic Cost Optimization (t=0.07) & 3-Tiered Routing
                                 │
      ┌──────────────────────────┼──────────────────────────┐
      ▼                          ▼                          ▼
LOW RISK (p < 0.03)      MEDIUM RISK (0.03 <= p < 0.30) HIGH RISK (p >= 0.30)
  Action: ALLOW             Action: CHALLENGE / 2FA       Action: HARD_BLOCK
(80.6% Vol / $8.8M GMV)     (17.2% Vol / Step-Up OTP)     (2.2% Vol / 67.4% Prec)
```

### 1. Three-Tiered Operational Risk Zones
To prevent GMV destruction from binary hard-blocking, incoming transactions are dynamically routed:
* **LOW RISK (`p < 0.03`) $\rightarrow$ `ALLOW`**: Frictionless checkout for 80.58% of volume ($8.82M GMV).
* **MEDIUM RISK (`0.03 <= p < 0.30`) $\rightarrow$ `CHALLENGE / STEP-UP 2FA`**: Triggers 2FA / OTP check. Honest cardholders complete verification to save GMV, while fraudsters drop off.
* **HIGH RISK (`p >= 0.30`) $\rightarrow$ `HARD_BLOCK`**: Immediate rejection for severe fraud threats (67.38% Precision).

### 2. Dynamic Value-Based Financial Cost Model
Calculates exact financial loss accounting for transaction amounts, merchant margins, churn penalties, and chargeback fees:
$$\text{False Positive Cost (FP)} = (\text{TransactionAmt} \times \text{Merchant\_Margin}) + \text{Churn\_Penalty}$$
$$\text{False Negative Cost (FN)} = \text{TransactionAmt} + \text{Chargeback\_Processor\_Fee}$$
*(Defaults: `Merchant_Margin = 0.03`, `Churn_Penalty = $10.00`, `Chargeback_Fee = $15.00`)*.

### 3. SHAP Operational Alert Extraction (`explain_transaction_alert`)
Converts raw float SHAP importance scores into actionable dashboard alerts for risk analysts:
* *Example Alert Output*: `[!] Extreme 24-Hour Velocity (5 tx in last 24 hours)`, `[!] High Email Domain Diversity (2.50 domain diversity ratio)`, `[!] Unusual Amount for Card Profile (3.45 z-score relative to card mean)`.

---

## 📈 Model Performance Benchmarks

### Sealed Test Set Metrics

| Model / Blend | Sealed Test ROC-AUC | Sealed Test PR-AUC |
| :--- | :---: | :---: |
| Logistic Regression | 0.8272 | 0.1761 |
| PyTorch MLP Neural Network | 0.8271 | 0.1922 |
| Baseline XGBoost (No Velocity) | 0.8900 | 0.5026 |
| Calibrated XGBoost + Velocity + Diversity | **0.8929** | **0.5100** |
| **Stacked Ensemble Blend (Final)** | **0.8884** | **0.5116** |

---

## 🔍 SHAP Feature Importance Rankings

Global SHAP analysis highlights top feature drivers across 431 active features:

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
├── fraud_detection.py          # Main End-to-End Pipeline Script
├── score_batch.py              # CLI Batch Inference Script (with operational_action)
├── predict_transaction.py      # Interactive Real-Time Simulator (ALLOW/CHALLENGE/HARD_BLOCK)
├── fraud_detection.ipynb       # Project Jupyter Notebook
├── README.md                   # Documentation
└── requirements.txt            # Dependencies
```

### Running the Pipeline & Tools

```bash
# Run full pipeline with dynamic cost model & 3-tiered action routing
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
