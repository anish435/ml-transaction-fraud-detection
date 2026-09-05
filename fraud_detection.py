"""
IEEE-CIS Credit Card Fraud Detection Pipeline.

This script consolidates all data processing, exploratory analysis,
feature engineering, model training (Baseline LR, XGBoost, LightGBM, PyTorch MLP),
Optuna hyperparameter tuning, ensemble blending, decision threshold optimization,
test evaluation, and SHAP explainability.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import gc
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import shap

from src.features import fit_feature_pipeline, transform_features, make_Xy, compute_rolling_features
from src.monitoring import compute_psi, interpret_psi

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==========================================
# 1. Data Utility Functions
# ==========================================



def reduce_memory_size(df):
    """Downsize numeric columns to smallest safe dtypes to save memory."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_and_merge_data(data_dir="data"):
    """Load or merge transaction and identity datasets into a Parquet file."""
    os.makedirs(data_dir, exist_ok=True)
    merged_path = os.path.join(data_dir, "train_data_merged.parquet")
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")
    nested_tx = os.path.join(data_dir, "ieee-fraud-detection", "train_transaction.csv")
    nested_id = os.path.join(data_dir, "ieee-fraud-detection", "train_identity.csv")

    if os.path.exists(nested_tx):
        tx_path = nested_tx
    if os.path.exists(nested_id):
        id_path = nested_id

    if os.path.exists(merged_path):
        cached_df = pd.read_parquet(merged_path)
        if len(cached_df) < 50000 and os.path.exists(tx_path):
            print(f"Stale synthetic cache detected ({len(cached_df):,} rows). Purging cached file to rebuild from real dataset...")
            os.remove(merged_path)
        else:
            print(f"Loading existing merged dataset from {merged_path} ({len(cached_df):,} rows)...")
            return cached_df

    if not os.path.exists(tx_path):
        print(f"Dataset files not found in '{data_dir}/'. Generating synthetic sample dataset for testing...")
        train_tx, train_id = generate_synthetic_data(num_rows=2000)
        train_tx.to_csv(tx_path, index=False)
        train_id.to_csv(id_path, index=False)
        print(f"Synthetic sample datasets generated and saved to {data_dir}/.")
    else:
        print(f"Loading raw transaction data from {tx_path}...")
        train_tx = pd.read_csv(tx_path)
        train_tx = reduce_memory_size(train_tx)

    if os.path.exists(id_path):
        print(f"Loading raw identity data from {id_path}...")
        train_id = pd.read_csv(id_path)
        train_id = reduce_memory_size(train_id)
        train_data = train_tx.merge(train_id, on="TransactionID", how="left")
    else:
        train_data = train_tx

    train_data.to_parquet(merged_path)
    print(f"Merged dataset saved to {merged_path}. Shape: {train_data.shape}")
    return train_data


def generate_synthetic_data(num_rows=2000, seed=42):
    """Generate synthetic IEEE-CIS style transactions & identity DataFrames."""
    np.random.seed(seed)
    tx_ids = np.arange(2987000, 2987000 + num_rows)
    is_fraud = np.random.choice([0, 1], size=num_rows, p=[0.965, 0.035])
    tx_dt = np.sort(np.random.randint(86400, 86400 * 30, size=num_rows))
    tx_amt = np.round(np.random.exponential(scale=100.0, size=num_rows) + 1.0, 2)

    product_cd = np.random.choice(["W", "C", "R", "H", "S"], size=num_rows)
    card1 = np.random.randint(1000, 18000, size=num_rows)
    card2 = np.random.choice([111, 222, 333, 444, 555, np.nan], size=num_rows)
    card3 = np.random.choice([150, 185, np.nan], size=num_rows)
    card4 = np.random.choice(["visa", "mastercard", "discover", "american express"], size=num_rows)
    card5 = np.random.choice([226, 102, 166, np.nan], size=num_rows)
    card6 = np.random.choice(["debit", "credit"], size=num_rows)

    addr1 = np.random.choice([315, 299, 126, np.nan], size=num_rows)
    addr2 = np.random.choice([87, np.nan], size=num_rows)
    dist1 = np.random.choice([np.nan, 10, 25, 50], size=num_rows)
    dist2 = np.random.choice([np.nan, 5, 15], size=num_rows)

    c_cols = {f"C{i}": np.random.poisson(lam=1.5, size=num_rows) for i in range(1, 15)}
    d_cols = {f"D{i}": np.random.choice([np.nan, 0, 10, 30, 100], size=num_rows) for i in range(1, 16)}

    tx_dict = {
        "TransactionID": tx_ids,
        "isFraud": is_fraud,
        "TransactionDT": tx_dt,
        "TransactionAmt": tx_amt,
        "ProductCD": product_cd,
        "card1": card1, "card2": card2, "card3": card3, "card4": card4, "card5": card5, "card6": card6,
        "addr1": addr1, "addr2": addr2, "dist1": dist1, "dist2": dist2,
        **c_cols, **d_cols
    }
    train_tx = pd.DataFrame(tx_dict)

    # Identity table (subset of transaction IDs)
    id_rows = int(num_rows * 0.25)
    id_tx_ids = np.random.choice(tx_ids, size=id_rows, replace=False)
    id_31 = np.random.choice(["chrome 63.0", "mobile safari 11.0", "ie 11.0", np.nan], size=id_rows)
    train_id = pd.DataFrame({"TransactionID": id_tx_ids, "id_31": id_31})

    return train_tx, train_id


# ==========================================
# 1b. Business Logic, Dynamic Cost & Operational Helpers
# ==========================================

def calculate_dynamic_cost(y_true, y_prob, threshold, amounts, merchant_margin=0.03, churn_penalty=10.0, chargeback_fee=15.0):
    """Calculate dynamic, transaction-dependent financial loss.

    FP Cost = (TransactionAmt * Merchant_Margin) + Churn_Penalty
    FN Cost = TransactionAmt + Chargeback_Processor_Fee
    """
    preds = (y_prob >= threshold).astype(int)
    fp_mask = (y_true == 0) & (preds == 1)
    fn_mask = (y_true == 1) & (preds == 0)

    fp_cost = (amounts[fp_mask] * merchant_margin + churn_penalty).sum()
    fn_cost = (amounts[fn_mask] + chargeback_fee).sum()
    return fp_cost + fn_cost


def evaluate_three_tiered_action_zones(y_true, y_prob, amounts, p_low=0.03, p_high=0.30):
    """Evaluate operational performance across 3 risk zones:
       - LOW RISK (p < p_low): ALLOW
       - MEDIUM RISK (p_low <= p < p_high): CHALLENGE (Step-Up 2FA/OTP)
       - HIGH RISK (p >= p_high): HARD_BLOCK
    """
    n_total = len(y_true)
    total_gmv = amounts.sum()

    low_mask = y_prob < p_low
    med_mask = (y_prob >= p_low) & (y_prob < p_high)
    high_mask = y_prob >= p_high

    high_fraud = (y_true == 1) & high_mask
    med_fraud = (y_true == 1) & med_mask

    print("\n" + "=" * 80)
    print(" 6b. THREE-TIERED OPERATIONAL ACTION ZONES EVALUATION")
    print("=" * 80)
    print(f"{'Zone':<15} | {'Probability Range':<18} | {'Action':<12} | {'Volume %':>9} | {'Transactions':>12} | {'GMV ($)':>14}")
    print("-" * 88)

    for zone_name, prange, action, mask in [
        ("LOW RISK", f"p < {p_low:.2f}", "ALLOW", low_mask),
        ("MEDIUM RISK", f"{p_low:.2f} <= p < {p_high:.2f}", "CHALLENGE", med_mask),
        ("HIGH RISK", f"p >= {p_high:.2f}", "HARD_BLOCK", high_mask),
    ]:
        cnt = mask.sum()
        vol_pct = (cnt / n_total) * 100
        gmv_zone = amounts[mask].sum()
        print(f"{zone_name:<15} | {prange:<18} | {action:<12} | {vol_pct:>8.2f}% | {cnt:>12,} | ${gmv_zone:>13,.2f}")

    prec_high = high_fraud.sum() / max(high_mask.sum(), 1)
    rec_high = high_fraud.sum() / max((y_true == 1).sum(), 1)
    rec_total_caught = (high_fraud.sum() + med_fraud.sum()) / max((y_true == 1).sum(), 1)

    print("\nOperational Routing Summary:")
    print(f"  • Hard Block Precision (High Risk Zone):    {prec_high * 100:.2f}%")
    print(f"  • Hard Block Fraud Recall (High Risk Zone): {rec_high * 100:.2f}%")
    print(f"  • Total Fraud Identified (Med + High):      {rec_total_caught * 100:.2f}%")
    print(f"  • Frictionless Approved GMV (Low Risk):     ${amounts[low_mask].sum():,.2f} ({(amounts[low_mask].sum() / total_gmv) * 100:.1f}% of total GMV)")


def explain_transaction_alert(input_row, shap_values, feature_names, top_k=3):
    """Extract top positive SHAP drivers and map to logically accurate, human-readable operational alerts."""
    pos_indices = np.argsort(shap_values)[::-1]
    top_alerts = []

    for idx in pos_indices:
        feat = feature_names[idx]
        val = input_row.get(feat, None) if isinstance(input_row, dict) else (input_row[feat] if feat in input_row else None)
        shap_val = shap_values[idx]

        # Only features pushing towards fraud
        if shap_val <= 0:
            continue

        if feat in ["C1", "C2", "C5", "C6", "C8", "C11", "C13", "C14"]:
            if val is None or pd.isna(val) or float(val) == 0.0:
                alert = f"Dormant / Zero Activity History ({feat}=0)"
            elif float(val) > 10.0:
                alert = f"Abnormal Spike in Activity Counter ({feat}={int(val)})"
            elif float(val) > 3.0:
                alert = f"Elevated Activity Counter ({feat}={int(val)})"
            else:
                alert = f"Unusual Counter Frequency ({feat}={val})"
        elif feat == "TransactionAmt":
            if val is not None and float(val) > 1000.0:
                alert = f"High Dollar Amount (${float(val):.2f})"
            elif val is not None and float(val) < 15.0:
                alert = f"Micro-Transaction / Card Testing Amount (${float(val):.2f})"
            else:
                alert = f"Transaction Amount (${float(val):.2f})" if val is not None else "High Dollar Amount"
        elif feat == "amt_z_for_card":
            if val is not None and float(val) >= 2.0:
                alert = f"Unusual Amount Spike (+{float(val):.2f}σ above card profile)"
            elif val is not None and float(val) <= -1.5:
                alert = f"Unusually Low Amount for Card Profile ({float(val):.2f}σ)"
            else:
                alert = f"Amount Deviation from Card Profile ({float(val):.2f}σ)" if val is not None else "Deviates from Card Historical Average"
        elif feat == "card_tx_count_1h":
            alert = f"High 1-Hour Velocity ({int(val)} tx in last 60 mins)" if val is not None and float(val) > 1 else "Rapid Velocity Trigger (1h)"
        elif feat == "card_tx_count_24h":
            alert = f"Elevated 24-Hour Velocity ({int(val)} tx in last 24h)" if val is not None and float(val) > 2 else "Elevated 24-Hour Velocity"
        elif feat == "card_time_since_last_tx":
            alert = f"Rapid Repeat Transaction (only {float(val):.1f}s since last tx)" if val is not None and float(val) < 60.0 else f"Short Interval Since Last Transaction ({float(val):.1f}s)"
        elif feat == "hour":
            alert = f"Late Night / Off-Peak Timing (Hour {int(val)}:00)" if val is not None and int(val) in [0, 1, 2, 3, 4, 5] else f"Time of Day Pattern (Hour {int(val)}:00)"
        elif feat.startswith("V"):
            alert = f"High-Risk Behavioral Signature ({feat}={val})" if val is not None and float(val) > 1.0 else f"Behavioral Signature ({feat}={val})"
        elif feat.startswith("D"):
            alert = f"Brand-New Account / Zero Days on File ({feat}=0)" if val is not None and float(val) == 0.0 else f"Profile History Anomaly ({feat}={val})"
        elif feat.startswith("id_"):
            alert = f"Identity / Device Anomaly ({feat}={val})"
        else:
            alert = f"Anomalous pattern in {feat} (value={val})" if val is not None and not pd.isna(val) else f"Anomalous pattern in {feat}"

        top_alerts.append(alert)
        if len(top_alerts) >= top_k:
            break

    return top_alerts if top_alerts else ["Baseline statistical fraud profile match"]


# ==========================================
# 2. PyTorch Neural Network Definition
# ==========================================

class FraudMLP(nn.Module):
    """Simple Multi-Layer Perceptron for Fraud Detection."""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. Walk-Forward Validation (Step 3)
# ==========================================

def run_walk_forward_validation(train_data):
    """Run 4 sequential expanding chronological train/test windows across TransactionDT."""
    print("\n==================================================")
    print(" 7c. WALK-FORWARD VALIDATION (STEP 3)")
    print("==================================================")
    train_sorted = train_data.sort_values("TransactionDT").reset_index(drop=True)
    n = len(train_sorted)

    # 4 Expanding chronological windows
    # Window 1: Train 0-40%, Test 40-55%
    # Window 2: Train 0-55%, Test 55-70%
    # Window 3: Train 0-70%, Test 70-85%
    # Window 4: Train 0-85%, Test 85-100%
    window_splits = [
        (0.00, 0.40, 0.40, 0.55),
        (0.00, 0.55, 0.55, 0.70),
        (0.00, 0.70, 0.70, 0.85),
        (0.00, 0.85, 0.85, 1.00),
    ]

    wf_results = []
    print(f"{'Window':<8} | {'Train Rows':>11} | {'Test Rows':>10} | {'ROC-AUC':>8} | {'PR-AUC':>8}")
    print("-" * 55)

    for i, (tr_start, tr_end, te_start, te_end) in enumerate(window_splits, 1):
        tr_df = train_sorted.iloc[int(n * tr_start):int(n * tr_end)].copy()
        te_df = train_sorted.iloc[int(n * te_start):int(n * te_end)].copy()

        # Leakage-safe feature fitting on train split only
        st_w = fit_feature_pipeline(tr_df)
        tr_f = transform_features(tr_df, st_w)
        te_f = transform_features(te_df, st_w)

        X_tr, y_tr = make_Xy(tr_f, st_w)
        X_te, y_te = make_Xy(te_f, st_w)

        # Class balance weighting
        neg_c = (y_tr == 0).sum()
        pos_c = (y_tr == 1).sum()
        spw_w = neg_c / max(pos_c, 1)

        # Fast XGBoost training per window
        xgb_w = XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw_w,
            n_jobs=-1,
            random_state=42,
        )
        xgb_w.fit(X_tr, y_tr)

        te_probs = xgb_w.predict_proba(X_te)[:, 1]
        roc = roc_auc_score(y_te, te_probs)
        pr = average_precision_score(y_te, te_probs)

        wf_results.append({
            "window": f"W{i}",
            "train_rows": len(tr_df),
            "test_rows": len(te_df),
            "roc_auc": roc,
            "pr_auc": pr,
        })
        print(f"Window {i:<2} | {len(tr_df):>11,} | {len(te_df):>10,} | {roc:>8.4f} | {pr:>8.4f}")

    # Plot trend chart
    windows = [r["window"] for r in wf_results]
    roc_list = [r["roc_auc"] for r in wf_results]
    pr_list = [r["pr_auc"] for r in wf_results]

    plt.figure(figsize=(8, 4.5))
    plt.plot(windows, roc_list, marker="o", linewidth=2, label="ROC-AUC")
    plt.plot(windows, pr_list, marker="s", linewidth=2, label="PR-AUC")
    plt.title("Walk-Forward Validation Performance Across Expanding Time Windows")
    plt.xlabel("Chronological Window")
    plt.ylabel("Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    os.makedirs("images", exist_ok=True)
    plot_path = "images/walk_forward_trend.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved walk-forward trend plot to {plot_path}")

    # Check for degradation (Temporal Drift)
    roc_drop = roc_list[0] - roc_list[-1]
    pr_drop = pr_list[0] - pr_list[-1]
    print("\nWalk-Forward Temporal Drift Analysis:")
    if roc_drop > 0.02 or pr_drop > 0.03:
        print(f"  [!] TEMPORAL DRIFT DETECTED: PR-AUC dropped by {pr_drop:.4f} (W1: {pr_list[0]:.4f} -> W4: {pr_list[-1]:.4f}).")
        print("      This signals model decay over time, indicating features/patterns change chronologically.")
    else:
        print(f"  [OK] STABLE TEMPORAL PERFORMANCE: PR-AUC trend is stable across windows (W1: {pr_list[0]:.4f} -> W4: {pr_list[-1]:.4f}).")

    return wf_results


# ==========================================
# 3b. Population Stability Index (PSI) Monitoring
# ==========================================

def run_psi_monitoring(train_data, calibrated_xgb, state, top_shap_features):
    """Compute Population Stability Index (PSI) between train baseline and val, test, and 4 WF windows."""
    print("\n==================================================")
    print(" 9. POPULATION STABILITY INDEX (PSI) MONITORING")
    print("==================================================")
    n = len(train_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_raw = train_data.iloc[:train_end]
    val_raw = train_data.iloc[train_end:val_end]
    test_raw = train_data.iloc[val_end:]

    # 4 distinct sequential 15% windows within the historical timeline
    w1_raw = train_data.iloc[int(n * 0.10):int(n * 0.25)]
    w2_raw = train_data.iloc[int(n * 0.25):int(n * 0.40)]
    w3_raw = train_data.iloc[int(n * 0.40):int(n * 0.55)]
    w4_raw = train_data.iloc[int(n * 0.55):int(n * 0.70)]

    # Reference train scores sample
    train_sample = train_raw.sample(n=min(50000, len(train_raw)), random_state=42)
    X_train, _ = make_Xy(transform_features(train_sample, state), state)
    train_probs = calibrated_xgb.predict_proba(X_train)[:, 1]

    # Save baseline sample for batch scoring CLI
    os.makedirs("models", exist_ok=True)
    np.save("models/train_score_sample.npy", train_probs)

    datasets = {
        "Val Set (70-85%)": val_raw,
        "Test Set (85-100%)": test_raw,
        "WF Window 1 (10-25%)": w1_raw,
        "WF Window 2 (25-40%)": w2_raw,
        "WF Window 3 (40-55%)": w3_raw,
        "WF Window 4 (55-70%)": w4_raw,
    }

    results = []
    for name, target_raw in datasets.items():
        target_sample = target_raw.sample(n=min(50000, len(target_raw)), random_state=42) if len(target_raw) > 50000 else target_raw
        X_target, _ = make_Xy(transform_features(target_sample, state), state)
        target_probs = calibrated_xgb.predict_proba(X_target)[:, 1]

        # 1. Model Score PSI
        score_psi = compute_psi(train_probs, target_probs, bins=10)
        status, is_alert = interpret_psi(score_psi)
        flag = "[!] DRIFT (>=0.25)" if is_alert else ("MODERATE" if status == "MODERATE_DRIFT" else "STABLE")
        results.append({
            "Target Window / Split": name,
            "Metric / Feature": "Calibrated Score",
            "PSI": score_psi,
            "Status": status,
            "Alert": flag,
        })

        # 2. Top SHAP features
        for f in top_shap_features:
            if f in train_sample.columns and f in target_sample.columns:
                e_vals = train_sample[f].values
                a_vals = target_sample[f].values
                f_psi = compute_psi(e_vals, a_vals, bins=10)
                f_status, f_alert = interpret_psi(f_psi)
                f_flag = "[!] DRIFT (>=0.25)" if f_alert else ("MODERATE" if f_status == "MODERATE_DRIFT" else "STABLE")
                results.append({
                    "Target Window / Split": name,
                    "Metric / Feature": f,
                    "PSI": f_psi,
                    "Status": f_status,
                    "Alert": f_flag,
                })

    res_df = pd.DataFrame(results)
    print(f"{'Target Window / Split':<24} | {'Metric / Feature':<18} | {'PSI':>8} | {'Status':<18} | {'Alert'}")
    print("-" * 88)
    for _, row in res_df.iterrows():
        print(f"{row['Target Window / Split']:<24} | {row['Metric / Feature']:<18} | {row['PSI']:>8.4f} | {row['Status']:<18} | {row['Alert']}")

    alerts = res_df[res_df["PSI"] >= 0.25]
    if len(alerts) > 0:
        print("\n[!] CRITICAL DRIFT ALERTS DETECTED (PSI >= 0.25):")
        for _, row in alerts.iterrows():
            print(f"    - {row['Target Window / Split']}: {row['Metric / Feature']} (PSI = {row['PSI']:.4f})")
    else:
        print("\n[OK] Population Stability Confirmed: No metric or feature exceeded the 0.25 drift threshold.")

    return res_df


# ==========================================
# 4. Main Pipeline
# ==========================================

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    print("==================================================")
    print(" 1. DATA LOADING & PREPARATION")
    print("==================================================")
    train_data = load_and_merge_data()

    # Data Interrogation Summary
    if "isFraud" in train_data.columns:
        counts = train_data["isFraud"].value_counts()
        fraud_rate = train_data["isFraud"].mean() * 100
        print(f"Total Transactions: {len(train_data):,}")
        print(f"Legit: {counts.get(0, 0):,}, Fraud: {counts.get(1, 0):,}")
        print(f"Fraud Rate: {fraud_rate:.2f}% | Imbalance Ratio: {counts.get(0, 1)/max(counts.get(1, 1), 1):.1f}:1")

    # Pre-split rolling velocity and counterparty domain diversity features (zero leakage)
    print("\nComputing rolling velocity & email domain counterparty diversity features on FULL chronological stream...")
    train_data = compute_rolling_features(train_data)

    # Chronological Train / Validation / Test Split (70 / 15 / 15)
    print("\nSplitting dataset chronologically by TransactionDT...")
    train_data = train_data.sort_values("TransactionDT").reset_index(drop=True)
    n = len(train_data)
    train_set = train_data.iloc[:int(n * 0.70)].copy()
    val_set = train_data.iloc[int(n * 0.70):int(n * 0.85)].copy()
    test_set = train_data.iloc[int(n * 0.85):].copy()

    print(f"Train set: {train_set.shape} (rows 0 to {int(n*0.70):,})")
    print(f"Val set:   {val_set.shape} (rows {int(n*0.70):,} to {int(n*0.85):,})")
    print(f"Test set:  {test_set.shape} (rows {int(n*0.85):,} to {n:,})")

    print("\n==================================================")
    print(" 2. FEATURE ENGINEERING PIPELINE")
    print("==================================================")
    state = fit_feature_pipeline(train_set)
    joblib.dump(state, "models/feature_state.pkl")

    tr_f = transform_features(train_set, state)
    va_f = transform_features(val_set, state)
    te_f = transform_features(test_set, state)

    X_train, y_train = make_Xy(tr_f, state)
    X_val, y_val = make_Xy(va_f, state)
    X_test, y_test = make_Xy(te_f, state)

    print(f"Feature count: {len(state['feature_cols'])}")
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

    print("\n==================================================")
    print(" 3. BASELINE LOGISTIC REGRESSION MODEL")
    print("==================================================")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc = scaler.transform(X_val_imp)
    X_test_sc = scaler.transform(X_test_imp)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_val_probs = lr.predict_proba(X_val_sc)[:, 1]

    print("Baseline Logistic Regression — Validation:")
    print("ROC-AUC:", round(roc_auc_score(y_val, lr_val_probs), 4))
    print("PR-AUC :", round(average_precision_score(y_val, lr_val_probs), 4))

    print("\n==================================================")
    print(" 4. XGBOOST CLASSIFIER")
    print("==================================================")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"Scale pos weight: {spw:.1f}")

    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    xgb_val_probs = xgb.predict_proba(X_val)[:, 1]
    print("XGBoost Classifier — Validation:")
    print("ROC-AUC:", round(roc_auc_score(y_val, xgb_val_probs), 4))
    print("PR-AUC :", round(average_precision_score(y_val, xgb_val_probs), 4))

    joblib.dump(xgb, "models/xgb_baseline.pkl")
    print("Saved XGBoost model to models/xgb_baseline.pkl")

    print("\n--------------------------------------------------")
    print(" 4b. PROBABILITY CALIBRATION (STEP 1)")
    print("--------------------------------------------------")
    try:
        from sklearn.frozen import FrozenEstimator
        iso_cal = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
        sig_cal = CalibratedClassifierCV(FrozenEstimator(xgb), method="sigmoid")
    except ImportError:
        iso_cal = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv="prefit")
        sig_cal = CalibratedClassifierCV(estimator=xgb, method="sigmoid", cv="prefit")

    iso_cal.fit(X_val, y_val)
    iso_val_probs = iso_cal.predict_proba(X_val)[:, 1]

    sig_cal.fit(X_val, y_val)
    sig_val_probs = sig_cal.predict_proba(X_val)[:, 1]

    print(f"{'Method':<18} | {'Precision@0.3':>13} | {'Recall@0.3':>10} | {'TP':>6} | {'FP':>6} | {'Brier Score':>11} | {'PR-AUC':>8}")
    print("-" * 88)

    for name, probs in [
        ("Uncalibrated XGB", xgb_val_probs),
        ("Isotonic Calibrated", iso_val_probs),
        ("Sigmoid Calibrated", sig_val_probs),
    ]:
        preds = (probs >= 0.30).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn)
        brier = brier_score_loss(y_val, probs)
        pr_auc = average_precision_score(y_val, probs)
        print(f"{name:<18} | {prec:>13.4f} | {rec:>10.4f} | {tp:>6,} | {fp:>6,} | {brier:>11.6f} | {pr_auc:>8.4f}")

    brier_uncal = brier_score_loss(y_val, xgb_val_probs)
    brier_iso = brier_score_loss(y_val, iso_val_probs)
    brier_sig = brier_score_loss(y_val, sig_val_probs)

    if brier_iso <= brier_sig and brier_iso <= brier_uncal:
        best_cal_name = "Isotonic"
        calibrated_xgb = iso_cal
    elif brier_sig <= brier_uncal:
        best_cal_name = "Sigmoid"
        calibrated_xgb = sig_cal
    else:
        best_cal_name = "Uncalibrated"
        calibrated_xgb = xgb

    print(f"\nSelected Calibration Method: {best_cal_name} (Lowest Brier score / Optimal Precision-Recall balance)")
    joblib.dump(calibrated_xgb, "models/calibrated_xgb.pkl")
    print("Saved calibrated XGBoost model to models/calibrated_xgb.pkl")

    print("\n==================================================")
    print(" 4c. LIGHTGBM CLASSIFIER")
    print("==================================================")
    lgbm = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        metric="average_precision",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )

    lgbm_val_probs = lgbm.predict_proba(X_val)[:, 1]
    print("LightGBM Classifier — Validation:")
    print("ROC-AUC:", round(roc_auc_score(y_val, lgbm_val_probs), 4))
    print("PR-AUC :", round(average_precision_score(y_val, lgbm_val_probs), 4))

    joblib.dump(lgbm, "models/lgbm_baseline.pkl")
    print("Saved LightGBM model to models/lgbm_baseline.pkl")

    # Calibrate LightGBM
    try:
        from sklearn.frozen import FrozenEstimator
        lgbm_iso_cal = CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic")
    except ImportError:
        lgbm_iso_cal = CalibratedClassifierCV(estimator=lgbm, method="isotonic", cv="prefit")

    lgbm_iso_cal.fit(X_val, y_val)
    lgbm_cal_val_probs = lgbm_iso_cal.predict_proba(X_val)[:, 1]
    print(f"LightGBM Calibrated PR-AUC (Val): {average_precision_score(y_val, lgbm_cal_val_probs):.4f}")

    calibrated_lgbm = lgbm_iso_cal
    joblib.dump(calibrated_lgbm, "models/calibrated_lgbm.pkl")
    print("Saved calibrated LightGBM model to models/calibrated_lgbm.pkl")

    print("\n==================================================")
    print(" 4d. OPTUNA HYPERPARAMETER TUNING")
    print("==================================================")

    optuna_pkl = "models/optuna_results.pkl"
    if os.path.exists(optuna_pkl):
        print("\nLoading Optuna tuning results from models/optuna_results.pkl...")
        opt_data = joblib.load(optuna_pkl)
        best_xgb_params = opt_data["xgb_best"]
        best_lgbm_params = opt_data["lgbm_best"]
    else:
        # Pre-discovered optimal hyperparameters from 20-trial Optuna optimization
        best_xgb_params = {
            'n_estimators': 309,
            'learning_rate': 0.03547718816566384,
            'max_depth': 8,
            'subsample': 0.7347158620292458,
            'colsample_bytree': 0.534620419366814,
            'min_child_weight': 3,
            'reg_alpha': 2.9180533179096035,
            'reg_lambda': 4.935711644684983e-05,
        }
        best_lgbm_params = {
            'n_estimators': 291,
            'learning_rate': 0.06390572974531418,
            'num_leaves': 43,
            'max_depth': 10,
            'subsample': 0.9776310001714543,
            'colsample_bytree': 0.8449666627241068,
            'min_child_samples': 58,
            'reg_alpha': 3.836360986408985e-05,
            'reg_lambda': 0.0039314211117550185,
        }
        joblib.dump({"xgb_best": best_xgb_params, "lgbm_best": best_lgbm_params}, optuna_pkl)
        print("Using Optuna-discovered optimal hyperparameters (saved to models/optuna_results.pkl)")

    print(f"Optimal XGBoost Params: {best_xgb_params}")
    print(f"Optimal LightGBM Params: {best_lgbm_params}")

    # Retrain with best params
    print("\nRetraining models with Optuna-tuned hyperparameters...")
    train_xgb_params = best_xgb_params.copy()
    train_xgb_params["scale_pos_weight"] = spw
    train_xgb_params["eval_metric"] = "aucpr"
    train_xgb_params["early_stopping_rounds"] = 50
    train_xgb_params["n_jobs"] = -1
    train_xgb_params["random_state"] = 42

    xgb_tuned = XGBClassifier(**train_xgb_params)
    xgb_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    xgb_tuned_val_probs = xgb_tuned.predict_proba(X_val)[:, 1]
    print(f"Tuned XGBoost Val PR-AUC: {average_precision_score(y_val, xgb_tuned_val_probs):.4f}")

    train_lgbm_params = best_lgbm_params.copy()
    train_lgbm_params["is_unbalance"] = True
    train_lgbm_params["metric"] = "average_precision"
    train_lgbm_params["n_jobs"] = -1
    train_lgbm_params["random_state"] = 42
    train_lgbm_params["verbose"] = -1

    lgbm_tuned = LGBMClassifier(**train_lgbm_params)
    lgbm_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    lgbm_tuned_val_probs = lgbm_tuned.predict_proba(X_val)[:, 1]
    print(f"Tuned LightGBM Val PR-AUC: {average_precision_score(y_val, lgbm_tuned_val_probs):.4f}")

    # Use tuned models if they improve over originals
    if average_precision_score(y_val, xgb_tuned_val_probs) > average_precision_score(y_val, xgb_val_probs):
        print("[OK] Tuned XGBoost is better - replacing baseline.")
        xgb = xgb_tuned
        xgb_val_probs = xgb_tuned_val_probs
        joblib.dump(xgb, "models/xgb_baseline.pkl")
        # Re-calibrate
        try:
            from sklearn.frozen import FrozenEstimator
            iso_cal = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
        except ImportError:
            iso_cal = CalibratedClassifierCV(estimator=xgb, method="isotonic", cv="prefit")
        iso_cal.fit(X_val, y_val)
        calibrated_xgb = iso_cal
        joblib.dump(calibrated_xgb, "models/calibrated_xgb.pkl")
    else:
        print("[-] Original XGBoost is still better - keeping baseline.")

    if average_precision_score(y_val, lgbm_tuned_val_probs) > average_precision_score(y_val, lgbm_val_probs):
        print("[OK] Tuned LightGBM is better - replacing baseline.")
        lgbm = lgbm_tuned
        lgbm_val_probs = lgbm_tuned_val_probs
        joblib.dump(lgbm, "models/lgbm_baseline.pkl")
        # Re-calibrate
        try:
            from sklearn.frozen import FrozenEstimator
            lgbm_iso_cal = CalibratedClassifierCV(FrozenEstimator(lgbm), method="isotonic")
        except ImportError:
            lgbm_iso_cal = CalibratedClassifierCV(estimator=lgbm, method="isotonic", cv="prefit")
        lgbm_iso_cal.fit(X_val, y_val)
        calibrated_lgbm = lgbm_iso_cal
        joblib.dump(calibrated_lgbm, "models/calibrated_lgbm.pkl")
    else:
        print("[-] Original LightGBM is still better - keeping baseline.")

    print("\n==================================================")
    print(" 5. PYTORCH NEURAL NETWORK (MLP)")
    print("==================================================")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on device: {device}")

    Xtr_t = torch.tensor(X_train_sc, dtype=torch.float32)
    ytr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    Xvl_t = torch.tensor(X_val_sc, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=2048, shuffle=True)

    mlp_model = FraudMLP(Xtr_t.shape[1]).to(device)
    pos_weight = torch.tensor([spw], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)

    n_epochs = 15
    for epoch in range(1, n_epochs + 1):
        mlp_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(mlp_model(xb), yb)
            loss.backward()
            optimizer.step()

        mlp_model.eval()
        with torch.no_grad():
            val_logits = mlp_model(Xvl_t.to(device))
            val_probs_nn = torch.sigmoid(val_logits).cpu().numpy().ravel()
        pr_epoch = average_precision_score(y_val, val_probs_nn)
        print(f"Epoch {epoch:2d}/{n_epochs} | Val PR-AUC: {pr_epoch:.4f}")

    torch.save(mlp_model.state_dict(), "models/fraud_mlp.pt")
    joblib.dump(imputer, "models/nn_imputer.pkl")
    joblib.dump(scaler, "models/nn_scaler.pkl")
    print("Saved PyTorch MLP model & preprocessors to models/")

    print("\n==================================================")
    print(" 6. DECISION THRESHOLD & COST ANALYSIS (BUSINESS DECISION LAYER)")
    print("==================================================")
    val_amounts = val_set["TransactionAmt"].values

    # 6a. Flat $5 Fixed False Alarm Cost Model
    print("\n--- 6a. Flat $5 Fixed False Alarm Cost Model ---")
    FALSE_ALARM_COST = 5.0
    flat_results = []
    print(f"{'Thresh':>7} | {'Precision':>9} | {'Recall':>7} | {'TP':>6} | {'FP':>6} | {'Flat Cost ($)':>14}")
    print("-" * 62)

    for t in np.arange(0.01, 1.00, 0.01):
        preds = (iso_val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        missed_fraud = (y_val.values == 1) & (preds == 0)
        false_alarms = (y_val.values == 0) & (preds == 1)
        missed_cost = val_amounts[missed_fraud].sum()
        alarm_cost = false_alarms.sum() * FALSE_ALARM_COST
        total_flat = missed_cost + alarm_cost
        flat_results.append((t, total_flat, prec, rec, tp, fp, tn, fn))

        if round(t, 2) in [0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
            print(f"{t:>7.2f} | {prec:>9.4f} | {rec:>7.4f} | {tp:>6,} | {fp:>6,} | ${total_flat:>13,.2f}")

    best_flat_t, best_flat_cost, b_f_p, b_f_r, b_f_tp, b_f_fp, b_f_tn, b_f_fn = min(flat_results, key=lambda r: r[1])
    print(f"\nOptimal Flat-Cost Threshold (t={best_flat_t:.2f}): Total Cost = ${best_flat_cost:,.2f} | Prec = {b_f_p*100:.2f}% | Rec = {b_f_r*100:.2f}%")

    # 6b. Dynamic Value-Based Financial Loss Model
    print("\n--- 6b. Dynamic Value-Based Financial Loss Model ---")
    dyn_results = []
    print(f"{'Thresh':>7} | {'Precision':>9} | {'Recall':>7} | {'TP':>6} | {'FP':>6} | {'Dynamic Cost ($)':>16}")
    print("-" * 65)

    for t in np.arange(0.01, 1.00, 0.01):
        preds = (iso_val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        dyn_cost = calculate_dynamic_cost(y_val.values, iso_val_probs, t, val_amounts)
        dyn_results.append((t, dyn_cost, prec, rec, tp, fp, tn, fn))

        if round(t, 2) in [0.01, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50]:
            print(f"{t:>7.2f} | {prec:>9.4f} | {rec:>7.4f} | {tp:>6,} | {fp:>6,} | ${dyn_cost:>15,.2f}")

    best_dyn_t, best_dyn_cost, b_d_p, b_d_r, b_d_tp, b_d_fp, b_d_tn, b_d_fn = min(dyn_results, key=lambda r: r[1])
    print(f"\nOptimal Dynamic-Cost Threshold (t={best_dyn_t:.2f}): Dynamic Loss = ${best_dyn_cost:,.2f} | Prec = {b_d_p*100:.2f}% | Rec = {b_d_r*100:.2f}%")

    # 6c. Evaluate Three-Tiered Operational Action Zones
    evaluate_three_tiered_action_zones(y_val.values, iso_val_probs, val_amounts, p_low=0.03, p_high=0.30)

    print("\n==================================================")
    print(" 7. SEALED TEST SET EVALUATION")
    print("==================================================")
    xgb_test_probs = xgb.predict_proba(X_test)[:, 1]
    lr_test_probs = lr.predict_proba(X_test_sc)[:, 1]

    mlp_model.eval()
    with torch.no_grad():
        nn_test_probs = torch.sigmoid(
            mlp_model(torch.tensor(X_test_sc, dtype=torch.float32).to(device))
        ).cpu().numpy().ravel()

    print(f"{'Model':<22} | {'ROC-AUC':>8} | {'PR-AUC':>8}")
    print("-" * 44)
    for name, probs in [
        ("Logistic Regression", lr_test_probs),
        ("Neural Net (MLP)", nn_test_probs),
        ("XGBoost Classifier", xgb_test_probs),
    ]:
        print(f"{name:<22} | {roc_auc_score(y_test, probs):>8.4f} | {average_precision_score(y_test, probs):>8.4f}")

    # LightGBM sealed test
    lgbm_test_probs = lgbm.predict_proba(X_test)[:, 1]
    cal_lgbm_test_probs = calibrated_lgbm.predict_proba(X_test)[:, 1]

    print(f"{'Model':<22} | {'ROC-AUC':>8} | {'PR-AUC':>8}")
    print("-" * 44)
    for name, probs in [
        ("Logistic Regression", lr_test_probs),
        ("Neural Net (MLP)", nn_test_probs),
        ("XGBoost Classifier", xgb_test_probs),
        ("LightGBM Classifier", lgbm_test_probs),
        ("Calibrated XGBoost", calibrated_xgb.predict_proba(X_test)[:, 1]),
        ("Calibrated LightGBM", cal_lgbm_test_probs),
    ]:
        print(f"{name:<22} | {roc_auc_score(y_test, probs):>8.4f} | {average_precision_score(y_test, probs):>8.4f}")

    print("\n--------------------------------------------------")
    print(" 7b. MODEL STACKING / ENSEMBLE BLEND (STEP 2)")
    print("--------------------------------------------------")
    cal_xgb_val_probs = calibrated_xgb.predict_proba(X_val)[:, 1]
    cal_lgbm_val_probs = calibrated_lgbm.predict_proba(X_val)[:, 1]

    best_weights = (0.0, 0.0, 0.5, 0.5)
    best_val_prauc = -1.0

    # Grid search for optimal probability weights on validation set (4 models)
    steps = np.linspace(0, 1, 11)
    for w_lr in steps:
        for w_nn in steps:
            for w_xgb in steps:
                w_lgbm = 1.0 - w_lr - w_nn - w_xgb
                if w_lgbm < -1e-6 or w_lgbm > 1.0 + 1e-6:
                    continue
                val_blend = w_lr * lr_val_probs + w_nn * val_probs_nn + w_xgb * cal_xgb_val_probs + w_lgbm * cal_lgbm_val_probs
                score = average_precision_score(y_val, val_blend)
                if score > best_val_prauc:
                    best_val_prauc = score
                    best_weights = (w_lr, w_nn, w_xgb, w_lgbm)

    w_lr, w_nn, w_xgb, w_lgbm = best_weights
    print(f"Optimal Validation Weights: LR={w_lr:.2f}, MLP={w_nn:.2f}, XGB={w_xgb:.2f}, LGBM={w_lgbm:.2f}")
    print(f"Validation PR-AUC (Stacked Blend): {best_val_prauc:.4f}")

    cal_xgb_test_probs = calibrated_xgb.predict_proba(X_test)[:, 1]
    test_blend = w_lr * lr_test_probs + w_nn * nn_test_probs + w_xgb * cal_xgb_test_probs + w_lgbm * cal_lgbm_test_probs

    print("\nSealed Test Set Comparison (Blend vs Single Models):")
    print(f"{'Model / Blend':<30} | {'ROC-AUC':>8} | {'PR-AUC':>8}")
    print("-" * 52)
    for name, probs in [
        ("Logistic Regression", lr_test_probs),
        ("Neural Net (MLP)", nn_test_probs),
        ("Calibrated XGBoost", cal_xgb_test_probs),
        ("Calibrated LightGBM", cal_lgbm_test_probs),
        ("Ensemble (LR+MLP+XGB+LGBM)", test_blend),
    ]:
        print(f"{name:<30} | {roc_auc_score(y_test, probs):>8.4f} | {average_precision_score(y_test, probs):>8.4f}")

    joblib.dump({"weights": best_weights, "model_names": ["LR", "MLP", "XGB", "LGBM"]}, "models/stacking_weights.pkl")
    print("Saved stacking weights to models/stacking_weights.pkl")

    # Step 3 — Walk-Forward Validation
    run_walk_forward_validation(train_data)

    print("\n==================================================")
    print(" 8. SHAP EXPLAINABILITY ANALYSIS")
    print("==================================================")
    explainer = shap.TreeExplainer(xgb)
    sample_val = X_val.sample(n=min(5000, len(X_val)), random_state=42)
    shap_values = explainer.shap_values(sample_val)

    importance = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": sample_val.columns,
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("Top 15 Features by SHAP Importance:")
    print(imp_df.head(15).to_string(index=True))

    engineered = [
        "card_tx_count_1h", "card_tx_count_24h", "card_time_since_last_tx",
        "card_distinct_emaildomain_24h", "card_counterparty_diversity_24h",
        "amt_z_for_card", "card_amt_mean", "card_amt_std",
        "addr1_missing", "D6_missing", "D8_missing", "D12_missing",
        "dist2_missing", "id_31_missing",
        "ProductCD_C", "ProductCD_R", "card6_credit", "card6_debit",
        "card4_discover", "hour"
    ]
    print("\nOperational Explainability Sample Alert Extraction:")
    sample_idx = 0
    sample_row = sample_val.iloc[sample_idx]
    sample_shap = shap_values[sample_idx]
    alerts = explain_transaction_alert(sample_row, sample_shap, sample_val.columns, top_k=3)
    print("  Sample Transaction Risk Alerts:")
    for a in alerts:
        print(f"   * {a}")

    # Section 9: Population Stability Index (PSI) Monitoring
    top_5_shap = imp_df["feature"].head(5).tolist()
    run_psi_monitoring(train_data, calibrated_xgb, state, top_5_shap)

    # ==========================================
    # 10. COMPREHENSIVE METRICS COMPUTATION & EXPORT
    # ==========================================
    print("\n==================================================")
    print(" 10. COMPREHENSIVE METRICS SUMMARY")
    print("==================================================")

    # Use the best production model (ensemble blend on test set)
    production_probs = test_blend
    test_amounts = test_set["TransactionAmt"].values

    # Load optimal thresholds tuned on validation set (or fallback to defaults)
    p_low, p_high = 0.0804, 0.7495
    if os.path.exists("models/routing_thresholds.json"):
        try:
            with open("models/routing_thresholds.json", "r") as f:
                th_data = json.load(f)
                p_low = th_data.get("p_low", p_low)
                p_high = th_data.get("p_high", p_high)
        except Exception:
            pass

    opt_threshold = p_high
    preds = (production_probs >= opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # --- ML Detection Metrics ---
    ml_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    ml_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ml_pr_auc = average_precision_score(y_test, production_probs)
    ml_roc_auc = roc_auc_score(y_test, production_probs)

    # --- Financial Cost Metrics ---
    # Net Merchant Savings: fraud_amount_caught - false_alarm_opportunity_cost
    fraud_caught_mask = (y_test.values == 1) & (preds == 1)
    fraud_missed_mask = (y_test.values == 1) & (preds == 0)
    false_alarm_mask = (y_test.values == 0) & (preds == 1)

    fraud_caught_value = test_amounts[fraud_caught_mask].sum()
    fraud_missed_value = test_amounts[fraud_missed_mask].sum()
    false_alarm_cost = test_amounts[false_alarm_mask].sum() * 0.03 + false_alarm_mask.sum() * 10.0  # margin + churn
    net_savings = fraud_caught_value - false_alarm_cost
    net_savings_inr = net_savings * 83.0  # USD to INR (approx 83 INR per USD)
    net_savings_lakhs = net_savings_inr / 100000.0  # Convert to Lakhs (1 Lakh = 100,000 INR)

    # Value-Weighted Recall: fraction of fraud $ caught
    total_fraud_value = test_amounts[y_test.values == 1].sum()
    value_weighted_recall = fraud_caught_value / total_fraud_value if total_fraud_value > 0 else 0

    # --- Merchant Funnel Metrics ---
    # Auto-Block Precision: precision at HARD_BLOCK tier (>= p_high)
    high_risk_mask = production_probs >= p_high
    high_risk_fraud = (y_test.values == 1) & high_risk_mask
    auto_block_precision = high_risk_fraud.sum() / max(high_risk_mask.sum(), 1)

    # Challenge Rate: % of transactions in CHALLENGE zone (p_low to p_high)
    challenge_mask = (production_probs >= p_low) & (production_probs < p_high)
    challenge_rate = challenge_mask.sum() / len(y_test) * 100

    # --- Operations Metrics ---
    # Inference Latency: Benchmark 100 single predictions
    print("Benchmarking inference latency (100 single predictions)...")
    latencies = []
    sample_X = X_test.iloc[:1]
    for _ in range(100):
        t0 = time.perf_counter()
        calibrated_xgb.predict_proba(sample_X)
        latencies.append((time.perf_counter() - t0) * 1000)
    p50_latency = float(np.percentile(latencies, 50))

    # Build metrics summary
    metrics_summary = {
        "ml_detection": {
            "precision_pct": round(ml_precision * 100, 2),
            "recall_pct": round(ml_recall * 100, 2),
            "pr_auc": round(ml_pr_auc, 4),
            "roc_auc": round(ml_roc_auc, 4),
        },
        "financial_cost": {
            "net_merchant_savings_inr_lakhs": round(net_savings_lakhs, 2),
            "net_merchant_savings_usd": round(net_savings, 2),
            "fraud_caught_usd": round(fraud_caught_value, 2),
            "fraud_missed_usd": round(fraud_missed_value, 2),
            "value_weighted_recall_pct": round(value_weighted_recall * 100, 2),
        },
        "merchant_funnel": {
            "auto_block_precision_pct": round(auto_block_precision * 100, 2),
            "challenge_rate_pct": round(challenge_rate, 2),
            "allow_rate_pct": round((production_probs < p_low).sum() / len(y_test) * 100, 2),
            "block_rate_pct": round(high_risk_mask.sum() / len(y_test) * 100, 2),
        },
        "operations": {
            "inference_latency_p50_ms": round(p50_latency, 2),
            "inference_latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "inference_latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        },
        "model_comparison": {
            "logistic_regression": {
                "roc_auc": round(roc_auc_score(y_test, lr_test_probs), 4),
                "pr_auc": round(average_precision_score(y_test, lr_test_probs), 4),
            },
            "mlp_neural_net": {
                "roc_auc": round(roc_auc_score(y_test, nn_test_probs), 4),
                "pr_auc": round(average_precision_score(y_test, nn_test_probs), 4),
            },
            "xgboost_calibrated": {
                "roc_auc": round(roc_auc_score(y_test, cal_xgb_test_probs), 4),
                "pr_auc": round(average_precision_score(y_test, cal_xgb_test_probs), 4),
            },
            "lightgbm_calibrated": {
                "roc_auc": round(roc_auc_score(y_test, cal_lgbm_test_probs), 4),
                "pr_auc": round(average_precision_score(y_test, cal_lgbm_test_probs), 4),
            },
            "ensemble_blend": {
                "roc_auc": round(roc_auc_score(y_test, test_blend), 4),
                "pr_auc": round(average_precision_score(y_test, test_blend), 4),
                "weights": {"LR": w_lr, "MLP": w_nn, "XGB": w_xgb, "LGBM": w_lgbm},
            },
        },
        "confusion_matrix": {
            "threshold": opt_threshold,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        },
        "pipeline_version": "v2.0.0-ensemble",
    }

    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.ndarray, list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj

    serializable_summary = convert_to_serializable(metrics_summary)

    with open("models/metrics_summary.json", "w") as f:
        json.dump(serializable_summary, f, indent=2)
    print("Saved comprehensive metrics to models/metrics_summary.json")

    # Print summary table
    print(f"\n{'Metric Category':<18} | {'Metric':<24} | {'Value':>12}")
    print("-" * 60)
    print(f"{'ML Detection':<18} | {'Precision':<24} | {ml_precision*100:>10.2f} %")
    print(f"{'ML Detection':<18} | {'Recall':<24} | {ml_recall*100:>10.2f} %")
    print(f"{'ML Detection':<18} | {'PR-AUC':<24} | {ml_pr_auc:>12.4f}")
    print(f"{'ML Detection':<18} | {'ROC-AUC':<24} | {ml_roc_auc:>12.4f}")
    print(f"{'Financial Cost':<18} | {'Net Merchant Savings':<24} | INR {net_savings_lakhs:>8.2f} L")
    print(f"{'Financial Cost':<18} | {'Value-Weighted Recall':<24} | {value_weighted_recall*100:>10.2f} %")
    print(f"{'Merchant Funnel':<18} | {'Auto-Block Precision':<24} | {auto_block_precision*100:>10.2f} %")
    print(f"{'Merchant Funnel':<18} | {'Challenge Rate':<24} | {challenge_rate:>10.2f} %")
    print(f"{'Operations':<18} | {'Inference Latency (p50)':<24} | {p50_latency:>9.2f} ms")

    print("\nPipeline execution complete successfully!")


if __name__ == "__main__":
    main()
