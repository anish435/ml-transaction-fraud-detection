# Credit Card Fraud Detection Pipeline — IEEE-CIS Benchmark

An end-to-end, production-ready machine learning framework for detecting credit card transaction fraud on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) benchmark (~590K transactions, 27.6:1 class imbalance). 

This project incorporates **leakage-safe feature engineering**, **past-only rolling velocity & counterparty domain diversity signals**, **isotonic probability calibration**, **ensemble model stacking**, **walk-forward temporal validation**, and **cost-minimizing threshold optimization**.

---

## 🚀 Key Highlights & Headline Results

* **Sealed Test PR-AUC**: **`0.5116`** (Stacked Blend) / **`0.5100`** (Calibrated XGBoost) vs ~0.035 no-skill baseline.
* **Sealed Test ROC-AUC**: **`0.8929`** (XGBoost + Velocity + Counterparty Diversity features).
* **Probability Calibration**: Isotonic calibration reduced probability Brier Loss from `0.0687` to **`0.0215`** (**-68.7% loss reduction**), cutting false alarms @ 0.30 threshold by **96.2%**.
* **Cost-Minimizing Threshold**: **`0.03`** (balancing transaction amount $ loss against a $5 false-alarm review penalty), minimizing total loss to **`$134,865.54`** while catching **82.74%** of all fraud.
* **Production Utilities**: Includes a zero-leakage batch inference script (`score_batch.py`) and an interactive CLI simulator (`predict_transaction.py`).

---

## 📊 Dataset Overview

The dataset merges transaction logs with identity/device metadata on `TransactionID`:

* **Total Transactions**: `590,540` rows (182-day time span)
* **Class Imbalance**: `569,877` Legitimate (96.50%) vs `20,663` Fraudulent (3.50%) $\rightarrow$ **`27.6 : 1`** ratio
* **Feature Schema**:
  * `TransactionDT`: Timedelta in seconds (used for chronological sorting & time-window features).
  * `TransactionAmt`: Transaction amount in USD ($).
  * `card1`–`card6`: Card identity, brand, type (debit/credit), category.
  * `addr1`, `addr2`: Billing region/zip codes.
  * `P_emaildomain`, `R_emaildomain`: Purchaser and recipient email domains.
  * `C1`–`C14`: Transaction counting features.
  * `D1`–`D15`: Timedeltas between past activity.
  * `V1`–`V339`: Vesta engineered interaction features.
  * `id_01`–`id_38`, `id_31`: Device, browser, and network identity metadata.

---

## 🛠️ System Architecture & Methodology

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
             Cost-Minimizing Threshold Optimization (t=0.03)
                                 │
           Production Scoring CLI (score_batch.py)
```

### 1. Leakage-Safe Chronological Splitting
Data is sorted strictly by `TransactionDT` into Train (70%), Validation (15%), and Test (15%). The test set remains sealed until final evaluation.

### 2. Rolling Velocity & Counterparty Diversity Features
All rolling features are computed on the **FULL chronological stream BEFORE train/val/test splitting** to guarantee continuous historical context across split boundaries:
* `card_tx_count_1h` & `card_tx_count_24h`: Transaction velocity for `card1` in the previous 1 hour and 24 hours.
* `card_time_since_last_tx`: Elapsed seconds since `card1`'s previous transaction.
* `card_distinct_emaildomain_24h`: Count of unique email domains (`P_emaildomain` / `R_emaildomain`) used by `card1` in 24 hours.
* `card_counterparty_diversity_24h`: $\frac{\text{card\_distinct\_emaildomain\_24h}}{\text{card\_tx\_count\_24h}}$ (domain diversity signal).

> **Zero Leakage**: Because every rolling calculation for transaction at index $i$ evaluates exclusively past timestamps ($t' \le t_i$), computing over the full stream maintains continuous history with **zero future leakage**.

### 3. Model Suite & Stacking
* **Logistic Regression**: Preprocessed with `StandardScaler` + `SimpleImputer` and balanced class weights.
* **PyTorch MLP (`FraudMLP`)**: 3-layer neural network trained with weighted `BCEWithLogitsLoss`.
* **XGBoost Classifier**: Hyperparameter-tuned tree model (`max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=27.4`).
* **Isotonic Calibration**: `CalibratedClassifierCV` fitting on `val_set` to align raw confidence scores with true empirical fraud rates.
* **Stacking Ensemble**: Probability-weighted blend (`0.00` LR + `0.05` MLP + `0.95` Calibrated XGBoost).

---

## 📈 Model Performance & Benchmarks

### Sealed Test Set Metrics

| Model / Blend | Sealed Test ROC-AUC | Sealed Test PR-AUC |
| :--- | :---: | :---: |
| Logistic Regression | 0.8272 | 0.1761 |
| PyTorch MLP Neural Network | 0.8271 | 0.1922 |
| Baseline XGBoost (No Velocity) | 0.8900 | 0.5026 |
| Calibrated XGBoost + Velocity + Diversity | **0.8929** | **0.5100** |
| **Stacked Ensemble Blend (Final)** | **0.8884** | **0.5116** |

### Calibrated Decision Threshold Cost Analysis

| Operating Threshold | Precision | Recall (Fraud Caught) | False Positives (FP) | Total Financial Loss ($) |
| :---: | :---: | :---: | :---: | :---: |
| `0.01` | 7.85% | 93.16% | 33,257 | $191,128.63 |
| **`0.03`** *(Optimal)* | **14.63%** | **82.74%** | **14,686** | **$134,865.54** |
| `0.05` | 21.31% | 73.87% | 8,296 | $158,690.08 |
| `0.10` | 33.49% | 64.20% | 3,878 | $185,911.55 |
| `0.30` | 67.38% | 43.20% | 636 | $300,467.75 |

---

## 🔍 SHAP Explainability Rankings

Global SHAP analysis highlights the top feature drivers across 431 active features:

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
├── score_batch.py              # CLI Batch Inference Script
├── predict_transaction.py      # Interactive Real-Time Simulator
├── fraud_detection.ipynb       # Project Jupyter Notebook
├── README.md                   # Documentation
└── requirements.txt            # Dependencies
```

### Installation

```bash
# Clone the repository
git clone https://github.com/anish435/ml-transaction-fraud-detection.git
cd ml-transaction-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
python fraud_detection.py
```

### Scoring New Transactions (Batch Inference CLI)

```bash
python score_batch.py --input data/new_transactions.csv --output data/scored.csv --threshold 0.03
```

### Running the Interactive Simulator CLI

```bash
python predict_transaction.py
```

---

## 📜 License & Acknowledgments

* Dataset provided by **IEEE Computational Intelligence Society (IEEE-CIS)** and **Vesta Corporation**.
* Built by [Anish](https://github.com/anish435).
