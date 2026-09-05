"""
Experiment: SMOTE vs scale_pos_weight for XGBoost, and PyTorch MLP Class-Weighting Analysis.

Evaluates on the untouched, sealed test set (85%-100% chronological split).
Production models in models/ are NOT modified.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

# Ensure repo root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features import compute_rolling_features, transform_features, make_Xy
from fraud_detection import FraudMLP

def main():
    print("=" * 80)
    print(" EXPERIMENT: SMOTE VS SCALE_POS_WEIGHT & PYTORCH MLP WEIGHTING")
    print("=" * 80)

    # 1. Load data and state
    print("\n1. Loading preprocessed data and feature pipeline state...")
    df = pd.read_parquet("data/train_data_merged.parquet")
    state = joblib.load("models/feature_state.pkl")

    # Chronological split matching production
    print("Computing rolling features on chronological stream...")
    df = compute_rolling_features(df)
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    n = len(df)
    train_set = df.iloc[:int(n * 0.70)].copy()
    val_set = df.iloc[int(n * 0.70):int(n * 0.85)].copy()
    test_set = df.iloc[int(n * 0.85):].copy()

    print(f"Train set: {len(train_set):,} | Val set: {len(val_set):,} | Sealed Test set: {len(test_set):,}")

    # Transform features
    print("Applying leakage-safe feature transformations...")
    tr_f = transform_features(train_set, state)
    va_f = transform_features(val_set, state)
    te_f = transform_features(test_set, state)

    X_train, y_train = make_Xy(tr_f, state)
    X_val, y_val = make_Xy(va_f, state)
    X_test, y_test = make_Xy(te_f, state)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f"Train Class Distribution: Legit={neg:,}, Fraud={pos:,} (Ratio={spw:.2f}:1)")
    print(f"Test Class Distribution:  Legit={(y_test==0).sum():,}, Fraud={(y_test==1).sum():,}")

    # =========================================================================
    # BASELINE EVALUATION: Current Production XGBoost (scale_pos_weight=27.6)
    # =========================================================================
    print("\n2. Evaluating Current Production XGBoost Model (scale_pos_weight=27.6)...")
    current_xgb = joblib.load("models/xgb_baseline.pkl")
    current_cal = joblib.load("models/calibrated_xgb.pkl")

    xgb_val_probs = current_xgb.predict_proba(X_val)[:, 1]
    xgb_test_probs = current_xgb.predict_proba(X_test)[:, 1]
    cal_test_probs = current_cal.predict_proba(X_test)[:, 1]

    val_roc_base = roc_auc_score(y_val, xgb_val_probs)
    val_pr_base = average_precision_score(y_val, xgb_val_probs)
    test_roc_base = roc_auc_score(y_test, xgb_test_probs)
    test_pr_base = average_precision_score(y_test, xgb_test_probs)

    test_roc_cal = roc_auc_score(y_test, cal_test_probs)
    test_pr_cal = average_precision_score(y_test, cal_test_probs)

    print(f"  Current XGB (Uncalibrated) -> Test ROC-AUC: {test_roc_base:.4f} | Test PR-AUC: {test_pr_base:.4f}")
    print(f"  Current XGB (Calibrated)   -> Test ROC-AUC: {test_roc_cal:.4f} | Test PR-AUC: {test_pr_cal:.4f}")

    # =========================================================================
    # EXPERIMENT 1: SMOTE on Training Set
    # =========================================================================
    print("\n3. Running SMOTE on Training Set...")
    print("   Note: SMOTE requires finite inputs (no NaNs), so SimpleImputer is fit on Train only.")
    imputer = SimpleImputer(strategy="median")
    t0_imp = time.perf_counter()
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    print(f"   Imputation completed in {time.perf_counter() - t0_imp:.2f}s.")

    t0_smote = time.perf_counter()
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_imp, y_train)
    smote_time = time.perf_counter() - t0_smote

    print(f"   SMOTE synthesis completed in {smote_time:.2f}s.")
    print(f"   Resampled Train Shape: {X_train_smote.shape}")
    print(f"   Resampled Distribution: Legit={(y_train_smote == 0).sum():,}, Fraud={(y_train_smote == 1).sum():,} (Balanced 1:1)")

    print("\n4. Training XGBoost on SMOTE-Resampled Training Set...")
    # Exact same hyperparameters as best model, with scale_pos_weight=1.0 since classes are balanced
    xgb_smote = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42,
    )

    t0_xgb = time.perf_counter()
    xgb_smote.fit(
        X_train_smote,
        y_train_smote,
        eval_set=[(X_val_imp, y_val)],
        verbose=100
    )
    xgb_smote_time = time.perf_counter() - t0_xgb
    print(f"   XGBoost (SMOTE) training completed in {xgb_smote_time:.2f}s. Best iteration: {xgb_smote.best_iteration}")

    # Evaluate SMOTE XGBoost
    smote_val_probs = xgb_smote.predict_proba(X_val_imp)[:, 1]
    smote_test_probs = xgb_smote.predict_proba(X_test_imp)[:, 1]

    val_roc_smote = roc_auc_score(y_val, smote_val_probs)
    val_pr_smote = average_precision_score(y_val, smote_val_probs)
    test_roc_smote = roc_auc_score(y_test, smote_test_probs)
    test_pr_smote = average_precision_score(y_test, smote_test_probs)

    print(f"   SMOTE XGBoost -> Val ROC-AUC: {val_roc_smote:.4f} | Val PR-AUC: {val_pr_smote:.4f}")
    print(f"   SMOTE XGBoost -> Test ROC-AUC: {test_roc_smote:.4f} | Test PR-AUC: {test_pr_smote:.4f}")

    # Save to experiments directory without touching production
    os.makedirs("experiments", exist_ok=True)
    joblib.dump(xgb_smote, "experiments/xgb_smote.pkl")
    print("   Saved experimental SMOTE model to experiments/xgb_smote.pkl.")

    # =========================================================================
    # EXPERIMENT 2: PyTorch MLP Class-Weighting Analysis
    # =========================================================================
    print("\n5. Analyzing PyTorch MLP Class-Weighting...")
    # Scale features for neural net
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc = scaler.transform(X_val_imp)
    X_test_sc = scaler.transform(X_test_imp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Training PyTorch MLPs on device: {device}")

    Xtr_t = torch.tensor(X_train_sc, dtype=torch.float32)
    ytr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    Xvl_t = torch.tensor(X_val_sc, dtype=torch.float32)
    Xte_t = torch.tensor(X_test_sc, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=2048, shuffle=True)

    def train_and_eval_mlp(use_pos_weight=False, name="MLP"):
        torch.manual_seed(42)
        model = FraudMLP(Xtr_t.shape[1]).to(device)

        if use_pos_weight:
            pw = torch.tensor([spw], dtype=torch.float32).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        n_epochs = 15

        for epoch in range(1, n_epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(Xvl_t.to(device))
            v_probs = torch.sigmoid(v_logits).cpu().numpy().ravel()
            t_logits = model(Xte_t.to(device))
            t_probs = torch.sigmoid(t_logits).cpu().numpy().ravel()

        v_roc = roc_auc_score(y_val, v_probs)
        v_pr = average_precision_score(y_val, v_probs)
        t_roc = roc_auc_score(y_test, t_probs)
        t_pr = average_precision_score(y_test, t_probs)

        return v_roc, v_pr, t_roc, t_pr

    print("   Training MLP Variant A: WITHOUT pos_weight (Unweighted BCE Loss)...")
    v_roc_mlp_unw, v_pr_mlp_unw, t_roc_mlp_unw, t_pr_mlp_unw = train_and_eval_mlp(use_pos_weight=False, name="Unweighted")
    print(f"   MLP Unweighted -> Test ROC-AUC: {t_roc_mlp_unw:.4f} | Test PR-AUC: {t_pr_mlp_unw:.4f}")

    print(f"   Training MLP Variant B: WITH pos_weight={spw:.2f} (Class-Weighted BCE Loss)...")
    v_roc_mlp_w, v_pr_mlp_w, t_roc_mlp_w, t_pr_mlp_w = train_and_eval_mlp(use_pos_weight=True, name="Weighted")
    print(f"   MLP Weighted   -> Test ROC-AUC: {t_roc_mlp_w:.4f} | Test PR-AUC: {t_pr_mlp_w:.4f}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 92)
    print(" EXPERIMENT RESULTS: SEALED TEST SET COMPARISON")
    print("=" * 92)
    print(f"{'Model Architecture':<24} | {'Imbalance Strategy':<22} | {'Test PR-AUC':>12} | {'Test ROC-AUC':>13} | {'Outcome':<12}")
    print("-" * 92)

    rows = [
        ("XGBoost (Production)", "scale_pos_weight=27.6", test_pr_base, test_roc_base, "BASELINE"),
        ("XGBoost (Experimental)", "SMOTE (1:1 Oversample)", test_pr_smote, test_roc_smote, "CANDIDATE"),
        ("PyTorch MLP", "Unweighted BCE Loss", t_pr_mlp_unw, t_roc_mlp_unw, "ABLATION"),
        ("PyTorch MLP", f"pos_weight={spw:.1f} (Current)", t_pr_mlp_w, t_roc_mlp_w, "CURRENT MLP"),
    ]

    for m_name, strat, pr, roc, out in rows:
        print(f"{m_name:<24} | {strat:<22} | {pr:>12.4f} | {roc:>13.4f} | {out:<12}")

    print("=" * 92)

    # Save summary results
    results_df = pd.DataFrame(rows, columns=["Model", "Imbalance_Strategy", "Test_PR_AUC", "Test_ROC_AUC", "Status"])
    results_df.to_csv("experiments/experiment_results.csv", index=False)
    print("Saved experiment results to experiments/experiment_results.csv.")

if __name__ == "__main__":
    main()
