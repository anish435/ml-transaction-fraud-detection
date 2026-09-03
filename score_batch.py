"""
Batch Fraud Scoring Script for Credit Card Transactions.

Usage:
    python score_batch.py --input new_transactions.csv --output scored.csv [--threshold 0.30]
"""

import argparse
import sys
import os
import joblib
import pandas as pd
import numpy as np

from src.features import transform_features, make_Xy, compute_rolling_features
from src.monitoring import compute_psi, interpret_psi


def parse_args():
    parser = argparse.ArgumentParser(description="Batch score credit card transactions for fraud risk.")
    parser.add_argument("--input", required=True, help="Path to input raw transactions CSV file")
    parser.add_argument("--output", required=True, help="Path where scored CSV will be saved")
    parser.add_argument("--threshold", type=float, default=0.30, help="Decision threshold (default: 0.30)")
    parser.add_argument("--check-psi", action="store_true", help="Compute Population Stability Index (PSI) drift against train baseline")
    return parser.parse_args()


def load_assets():
    state_path = "models/feature_state.pkl"
    calibrated_model_path = "models/calibrated_xgb.pkl"
    xgb_baseline_path = "models/xgb_baseline.pkl"

    if not os.path.exists(state_path):
        print(f"[ERROR] Saved feature pipeline state not found at '{state_path}'.")
        print("        Run 'python fraud_detection.py' first to build and save pipeline assets.")
        sys.exit(1)

    state = joblib.load(state_path)

    if os.path.exists(calibrated_model_path):
        print(f"Loading calibrated XGBoost model from '{calibrated_model_path}'...")
        model = joblib.load(calibrated_model_path)
    elif os.path.exists(xgb_baseline_path):
        print(f"[WARNING] Calibrated model not found. Loading baseline XGBoost model from '{xgb_baseline_path}'...")
        model = joblib.load(xgb_baseline_path)
    else:
        print("[ERROR] No trained model found in 'models/'. Run 'python fraud_detection.py' first.")
        sys.exit(1)

    return state, model


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"Reading input transactions from '{args.input}'...")
    raw_df = pd.read_csv(args.input)
    print(f"Loaded {len(raw_df):,} rows.")

    print("Computing rolling velocity & email domain counterparty diversity features...")
    raw_df = compute_rolling_features(raw_df)

    state, model = load_assets()

    # Apply frozen feature pipeline (NEVER refit — prevents data leakage)
    print("Transforming features using frozen train-set pipeline state...")
    df_transformed = transform_features(raw_df, state)
    X, _ = make_Xy(df_transformed, state)

    print(f"Evaluating fraud risk probabilities (Threshold = {args.threshold:.2f})...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= args.threshold).astype(int)

    # Construct result DataFrame preserving key input metadata
    output_df = raw_df.copy()
    output_df["fraud_probability"] = np.round(probs, 4)
    output_df["is_fraud_predicted"] = preds

    # Map to 3-Tiered Operational Action Zones
    conditions = [probs < 0.03, (probs >= 0.03) & (probs < 0.30), probs >= 0.30]
    actions = ["ALLOW", "CHALLENGE", "HARD_BLOCK"]
    output_df["operational_action"] = np.select(conditions, actions, default="ALLOW")

    # Reorder key columns to front if present
    front_cols = ["TransactionID", "TransactionDT", "TransactionAmt", "fraud_probability", "operational_action", "is_fraud_predicted"]
    ordered_cols = [c for c in front_cols if c in output_df.columns] + [c for c in output_df.columns if c not in front_cols]
    output_df = output_df[ordered_cols]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_df.to_csv(args.output, index=False)
    print(f"[OK] Scoring complete! Saved results to '{args.output}'.")
    print(f"     Total Scored : {len(output_df):,}")
    print(f"     Flagged Fraud: {(preds == 1).sum():,} ({(preds == 1).mean() * 100:.2f}%)")
    print(f"     Approved     : {(preds == 0).sum():,} ({(preds == 0).mean() * 100:.2f}%)")

    if args.check_psi:
        baseline_path = "models/train_score_sample.npy"
        if os.path.exists(baseline_path):
            baseline_scores = np.load(baseline_path)
            psi_val = compute_psi(baseline_scores, probs, bins=10)
            status, is_alert = interpret_psi(psi_val)
            print("\n--------------------------------------------------")
            print(" POPULATION STABILITY INDEX (PSI) MONITORING")
            print("--------------------------------------------------")
            print(f" Batch Score PSI vs Train Baseline: {psi_val:.4f} ({status})")
            if is_alert:
                print(" [!] ALERT: Significant distribution drift detected (PSI >= 0.25). Model retraining recommended!")
            elif status == "MODERATE_DRIFT":
                print(" [*] WARNING: Moderate distribution drift detected (0.10 <= PSI < 0.25). Monitor upcoming batches closely.")
            else:
                print(" [OK] Score distribution is STABLE (PSI < 0.10).")
        else:
            print("\n[NOTE] Baseline scores ('models/train_score_sample.npy') not found. Run 'python fraud_detection.py' to generate baseline.")


if __name__ == "__main__":
    main()
