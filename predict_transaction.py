"""
Interactive Fraud Detection Simulator.

Uses trained XGBoost model and leakage-safe feature pipeline
to predict fraud risk on individual user transacti2on inputs.
"""

import sys
import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.features import transform_features, make_Xy

THRESHOLD = 0.30


def load_model_assets():
    state_path = "models/feature_state.pkl"
    xgb_path = "models/xgb_baseline.pkl"

    if not os.path.exists(state_path) or not os.path.exists(xgb_path):
        print("❌ Error: Trained model assets not found in models/.")
        print("Please run 'python fraud_detection.py' first to train and save the model.")
        sys.exit(1)

    state = joblib.load(state_path)
    xgb = joblib.load(xgb_path)
    return state, xgb


def predict_single_transaction(input_dict, state, xgb):
    """Transform user input dict and return fraud probability and risk decision."""
    df = pd.DataFrame([input_dict])

    # Ensure required default columns exist in single transaction input
    default_cols = {
        "TransactionDT": input_dict.get("hour", 12) * 3600,
        "TransactionAmt": float(input_dict.get("TransactionAmt", 100.0)),
        "ProductCD": str(input_dict.get("ProductCD", "W")),
        "card1": int(input_dict.get("card1", 10000)),
        "card2": np.nan, "card3": np.nan, "card4": str(input_dict.get("card4", "visa")),
        "card5": np.nan, "card6": str(input_dict.get("card6", "debit")),
        "addr1": float(input_dict.get("addr1", 315)) if input_dict.get("addr1") else np.nan,
        "addr2": np.nan, "dist1": np.nan, "dist2": np.nan,
    }

    for col, val in default_cols.items():
        if col not in df.columns:
            df[col] = val

    # Fill C1-C14 and D1-D15 if missing
    for i in range(1, 15):
        if f"C{i}" not in df.columns:
            df[f"C{i}"] = 1.0
    for i in range(1, 16):
        if f"D{i}" not in df.columns:
            df[f"D{i}"] = np.nan

    # Device identity check
    if input_dict.get("has_identity", True):
        df["id_31"] = "chrome 63.0"
    else:
        df["id_31"] = np.nan

    # Apply feature pipeline
    df_transformed = transform_features(df, state)
    X, _ = make_Xy(df_transformed, state)

    prob = float(xgb.predict_proba(X)[0, 1])

    # Card amount z-score context
    card_mean = state["card_stats"].loc[state["card_stats"]["card1"] == df["card1"].iloc[0], "card_amt_mean"]
    mean_val = float(card_mean.iloc[0]) if len(card_mean) > 0 else state["global_mean"]

    return prob, mean_val, df_transformed


def display_result(prob, amount, mean_val):
    print("\n" + "=" * 55)
    print("           TRANSACTION EVALUATION RESULT           ")
    print("=" * 55)

    risk_pct = prob * 100
    print(f" Transaction Amount : ${amount:.2f}")
    print(f" Card Historical Avg: ${mean_val:.2f}")
    print(f" Fraud Probability  : {risk_pct:.2f}%")
    print("-" * 55)

    if prob < 0.03:
        print(" [*] OPERATIONAL ACTION: ALLOW")
        print(" [*] RISK LEVEL        : LOW RISK")
        print(" [*] REASON            : Smooth checkout path. Zero user friction.")
    elif prob < 0.30:
        print(" [!] OPERATIONAL ACTION: CHALLENGE / STEP-UP 2FA")
        print(" [!] RISK LEVEL        : MEDIUM RISK")
        print(" [!] REASON            : Trigger 2FA / OTP check to verify cardholder identity.")
    else:
        print(" [X] OPERATIONAL ACTION: HARD BLOCK")
        print(" [X] RISK LEVEL        : HIGH RISK DETECTED")
        print(" [X] REASON            : High-confidence fraud signature. Instantly rejected.")

    print("=" * 55 + "\n")


def run_preset_demo(state, xgb):
    print("\n--- PRESET DEMO SCENARIOS ---")
    print("1. Standard Grocery Purchase ($45.50, Debit Card, Daytime)")
    print("2. Suspicious High Amount Purchase ($1,499.00, Overnight 3 AM)")
    print("3. High-Risk Overseas Product Category 'C' ($450.00)")
    choice = input("Select a scenario (1-3): ").strip()

    if choice == "1":
        tx = {"TransactionAmt": 45.50, "ProductCD": "W", "card1": 13579, "card4": "visa", "card6": "debit", "hour": 14, "has_identity": False}
    elif choice == "2":
        tx = {"TransactionAmt": 1499.00, "ProductCD": "W", "card1": 13579, "card4": "mastercard", "card6": "credit", "hour": 3, "has_identity": True}
    elif choice == "3":
        tx = {"TransactionAmt": 450.00, "ProductCD": "C", "card1": 9876, "card4": "visa", "card6": "credit", "hour": 2, "has_identity": True}
    else:
        print("Invalid choice, defaulting to Scenario 1.")
        tx = {"TransactionAmt": 45.50, "ProductCD": "W", "card1": 13579, "card4": "visa", "card6": "debit", "hour": 14, "has_identity": False}

    prob, mean_val, _ = predict_single_transaction(tx, state, xgb)
    display_result(prob, tx["TransactionAmt"], mean_val)


def main():
    state, xgb = load_model_assets()

    print("==================================================")
    print(" 💳 REAL-TIME CREDIT CARD FRAUD DETECTOR")
    print("==================================================")
    print("Backend Model: XGBoost Classifier (IEEE-CIS Trained)")
    print("Operational Decision Threshold: 30.0% Risk Probability")

    while True:
        print("\nChoose an option:")
        print(" 1. Enter transaction details manually")
        print(" 2. Run preset test demo scenarios")
        print(" 3. Exit")

        mode = input("\nEnter choice (1, 2, or 3): ").strip()

        if mode == "3":
            print("\nExiting Fraud Detector. Have a great day!")
            break
        elif mode == "2":
            run_preset_demo(state, xgb)
            continue
        elif mode != "1":
            print("Invalid selection. Please enter 1, 2, or 3.")
            continue

        print("\n--- ENTER TRANSACTION DETAILS ---")
        try:
            amt_str = input("Transaction Amount ($) [e.g. 150.00]: ").strip()
            amount = float(amt_str) if amt_str else 150.00

            print("Product Category options: W (Web), C (Communication), R (Realtime), H (Host), S (Service)")
            product = input("Product Category [default: W]: ").strip().upper()
            if not product or product not in ["W", "C", "R", "H", "S"]:
                product = "W"

            card1_str = input("Card ID Number (card1) [e.g. 13579]: ").strip()
            card1 = int(card1_str) if card1_str else 13579

            card4 = input("Card Network (visa, mastercard, discover, american express) [default: visa]: ").strip().lower()
            if not card4:
                card4 = "visa"

            card6 = input("Card Type (debit or credit) [default: debit]: ").strip().lower()
            if not card6:
                card6 = "debit"

            hour_str = input("Hour of Day (0 - 23) [default: 14 for 2 PM]: ").strip()
            hour = int(hour_str) if hour_str.isdigit() and 0 <= int(hour_str) <= 23 else 14

            has_id_str = input("Is digital device/browser identity present? (y/n) [default: y]: ").strip().lower()
            has_id = has_id_str != "n"

            tx_input = {
                "TransactionAmt": amount,
                "ProductCD": product,
                "card1": card1,
                "card4": card4,
                "card6": card6,
                "hour": hour,
                "has_identity": has_id
            }

            prob, mean_val, _ = predict_single_transaction(tx_input, state, xgb)
            display_result(prob, amount, mean_val)

        except ValueError as e:
            print(f"❌ Input error: {e}. Please enter valid numbers.")


if __name__ == "__main__":
    main()
