"""
Leakage-Safe Feature Pipeline for Credit Card Fraud Detection.

This module provides the core data transformation and profiling logic for the fraud detection system.
Design mirrors scikit-learn's fit/transform split:

Mathematical Transformations & Profile Learning:
-----------------------------------------------
1. Card Profile Historical Statistics (Learned on TRAIN ONLY):
   - Grouping by `card1` to aggregate historical transaction amounts:
       mean(card1) = (1 / N) * sum(TransactionAmt)
       std(card1)  = sqrt( (1 / (N - 1)) * sum( (TransactionAmt - mean)^2 ) )
   - Global Fallbacks (handling cold-start instances for unseen cards in test/val):
       global_amt_mean = mean(TransactionAmt_train)
       global_amt_std  = std(TransactionAmt_train)

2. Runtime Amount Z-Score (`amt_z_for_card`):
   - Standardized deviation of current transaction amount relative to card's historical profile:
       amt_z_for_card = (TransactionAmt - card_amt_mean) / card_amt_std
   - Zero-Division Protection: If std == 0.0 (single purchase history or identical amounts),
     std is replaced with 1.0 (or global fallback std) to prevent Inf/NaN.

3. Pre-Split Chronological Rolling Features (Zero Leakage):
   - `card_time_since_last_tx`: Seconds elapsed since previous transaction for card1.
   - `card_tx_count_1h`: Transaction count in previous 3,600s window.
   - `card_tx_count_24h`: Transaction count in previous 86,400s window.
   - `card_distinct_emaildomain_24h`: Unique email domains (P_emaildomain/R_emaildomain) in 24h.
   - `card_counterparty_diversity_24h`: Ratio of distinct email domains to 24h count.

Usage:
    state = fit_feature_pipeline(train_df)
    X_train, y_train = make_Xy(transform_features(train_df, state), state)
    X_val,   y_val   = make_Xy(transform_features(val_df,   state), state)
    X_test,  y_test  = make_Xy(transform_features(test_df,  state), state)
"""

import numpy as np
import pandas as pd

# Low-cardinality categoricals to one-hot encode
CAT_COLS = ["ProductCD", "card4", "card6"]

# Missing-indicators kept because they separate fraud in EDA
KEEP_INDICATORS = {
    "addr1_missing": "addr1",
    "D6_missing": "D6",
    "D8_missing": "D8",
    "D12_missing": "D12",
    "dist2_missing": "dist2",
    "id_31_missing": "id_31",
}

# Columns that must never be used as model input features
EXCLUDE = ["isFraud", "TransactionID", "TransactionDT"]


def compute_rolling_features(df):
    """Compute per-card1 rolling velocity and email domain counterparty diversity features.

    Must be run on the FULL chronologically-sorted dataset prior to train/val/test
    splitting to guarantee continuous history across split boundaries with zero leakage.

    Mathematical Operations:
    ------------------------
    - `card_time_since_last_tx` = TransactionDT[i] - TransactionDT[i-1] (per card1)
    - `card_tx_count_1h`        = count(t in [t_i - 3600, t_i])
    - `card_tx_count_24h`       = count(t in [t_i - 86400, t_i])
    - `card_counterparty_diversity_24h` = distinct_domains(24h) / card_tx_count_24h
    """
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    # 1. Time since last transaction for card1 (seconds)
    df["card_time_since_last_tx"] = df.groupby("card1")["TransactionDT"].diff().fillna(999999.0)

    # Prepare purchaser and recipient email domain columns if present
    p_col = df["P_emaildomain"] if "P_emaildomain" in df.columns else pd.Series(np.nan, index=df.index)
    r_col = df["R_emaildomain"] if "R_emaildomain" in df.columns else pd.Series(np.nan, index=df.index)

    temp_df = df[["card1", "TransactionDT"]].copy()
    temp_df["_P"] = p_col.values
    temp_df["_R"] = r_col.values

    def calc_group_rolling(sub_df):
        times = sub_df["TransactionDT"].values
        p_domains = sub_df["_P"].values
        r_domains = sub_df["_R"].values
        n = len(times)

        left_1h = np.searchsorted(times, times - 3600, side="left")
        left_24h = np.searchsorted(times, times - 86400, side="left")

        c_1h = np.arange(n) - left_1h + 1
        c_24h = np.arange(n) - left_24h + 1
        distinct_domains = np.zeros(n, dtype=int)

        for i in range(n):
            l24 = left_24h[i]
            p_slice = p_domains[l24:i+1]
            r_slice = r_domains[l24:i+1]
            valid_p = p_slice[~pd.isnull(p_slice)]
            valid_r = r_slice[~pd.isnull(r_slice)]
            unique_d = set(valid_p).union(set(valid_r))
            distinct_domains[i] = len(unique_d) if len(unique_d) > 0 else 0

        return pd.DataFrame({
            "card_tx_count_1h": c_1h,
            "card_tx_count_24h": c_24h,
            "card_distinct_emaildomain_24h": distinct_domains,
        }, index=sub_df.index)

    res = temp_df.groupby("card1", group_keys=False).apply(calc_group_rolling, include_groups=False)
    df = pd.concat([df, res], axis=1)
    df["card_counterparty_diversity_24h"] = df["card_distinct_emaildomain_24h"] / df["card_tx_count_24h"]

    return df


def fit_feature_pipeline(train_df):
    """Learn all historical profiling statistics strictly from training data.

    Returns a frozen `state` dictionary consumed by `transform_features` and `make_Xy`.

    Learned States:
    ---------------
    1. `card_stats`: DataFrame mapping `card1` -> `card_amt_mean` & `card_amt_std`.
    2. `global_mean` & `global_std`: Fallback scalars for unseen cold-start cards.
    3. `feature_cols`: Frozen schema list of active model features.
    """
    state = {}

    # Per-card amount statistics — LEARNED ON TRAIN ONLY (leakage-critical)
    stats = (
        train_df.groupby("card1")["TransactionAmt"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["card1", "card_amt_mean", "card_amt_std"]
    state["card_stats"] = stats
    state["global_mean"] = float(train_df["TransactionAmt"].mean())
    state["global_std"] = float(train_df["TransactionAmt"].std())

    # Freeze feature column schema from a transformed training sample
    transformed = transform_features(train_df, state, _defining_schema=True)
    numeric = transformed.select_dtypes(include=[np.number, "bool"]).columns
    feature_cols = [c for c in numeric if c not in EXCLUDE]
    state["feature_cols"] = list(dict.fromkeys(feature_cols))

    return state


def transform_features(df, state, _defining_schema=False):
    """Apply historical profiling states and feature engineering to any split/stream.

    Parameters:
    -----------
    df : DataFrame
        Raw or pre-split transaction records.
    state : dict
        Frozen state learned during `fit_feature_pipeline` (contains card statistics,
        global fallbacks, and feature column schema).
    _defining_schema : bool
        Used internally by `fit_feature_pipeline` to freeze schema before state initialization.

    Returns:
    --------
    DataFrame : Feature-engineered DataFrame.
    """
    df = df.copy()

    # 1. Time feature (hour of the DT-reference day)
    df["hour"] = ((df["TransactionDT"] / 3600) % 24).astype(int)

    # 2. Missing-indicators (stateless — pure null checks)
    for ind_col, src_col in KEEP_INDICATORS.items():
        if ind_col in df.columns:
            df.drop(columns=[ind_col], inplace=True)
        if src_col in df.columns:
            df[ind_col] = df[src_col].isnull().astype(int)

    # 3. Card-level amount z-score (stateful — uses TRAIN stats from `state`)
    for col in ["card_amt_mean", "card_amt_std", "amt_z_for_card"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Left-merge train-learned historical card statistics
    df = df.merge(state["card_stats"], on="card1", how="left")

    # Impute cold-start cards using train global fallbacks
    df["card_amt_mean"] = df["card_amt_mean"].fillna(state["global_mean"])

    # Zero-division protection: Replace 0.0 std with 1.0 (or global std fallback)
    raw_std = df["card_amt_std"].fillna(state["global_std"])
    safe_std = np.where(raw_std == 0.0, 1.0, raw_std)
    df["card_amt_std"] = safe_std

    # Compute Z-score relative to card profile
    df["amt_z_for_card"] = (df["TransactionAmt"] - df["card_amt_mean"]) / df["card_amt_std"]

    # 4. One-hot encode low-cardinality categoricals (NaN as its own category)
    df = pd.get_dummies(df, columns=CAT_COLS, dummy_na=True, dtype=int)

    if _defining_schema:
        return df

    target = df["isFraud"] if "isFraud" in df.columns else None

    # 5. Align to the frozen train schema (add missing dummy cols as 0)
    df = df.reindex(columns=state["feature_cols"], fill_value=0)

    if target is not None:
        df["isFraud"] = target.values

    return df


def make_Xy(df, state):
    """Return model-ready X (frozen feature columns) and y (target Series, or None)."""
    X = df[state["feature_cols"]]
    y = df["isFraud"] if "isFraud" in df.columns else None
    return X, y
