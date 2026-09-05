"""
generate_incoming_pool.py — creates a pool of pre-generated synthetic
"future" disputes for the live-arrival simulation.

Does NOT touch:
    orders.csv
    disputes.csv
    golden_answers.csv

Output:
    <project_root>/data/incoming_disputes.csv

Schema matches what DisputePipeline needs:
order fields + dispute fields combined into one row.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 4242
N_INCOMING = 300

REASON_CODES = [
    "not_as_described",
    "item_not_received",
    "unauthorized_transaction",
    "duplicate_charge",
]

ITEM_CATEGORIES = [
    "apparel",
    "electronics",
    "groceries",
    "digital_goods",
    "home_goods",
]

MIN_DAYS_TO_DISPUTE = 1
MAX_DAYS_TO_DISPUTE = 24

START_INDEX = 100000
# Historical IDs are DSP-00000 ... DSP-14999
# Incoming IDs start at DSP-100000 to avoid collisions.


# ============================================================
# PROJECT PATHS
# ============================================================

# generate_incoming_pool.py is in the PROJECT ROOT.
#
# Example:
# D:\AI_Chargeback_Evidence_Responder\
#     generate_incoming_pool.py
#     app.py
#     pipeline.py
#     data\

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

OUT_PATH = DATA_DIR / "incoming_disputes.csv"


# ============================================================
# GENERATE POOL
# ============================================================

def generate_pool():

    rng = np.random.default_rng(SEED)

    rows = []

    for i in range(N_INCOMING):

        # ----------------------------------------------------
        # IDs
        # ----------------------------------------------------

        dispute_id = f"DSP-{START_INDEX + i:06d}"
        order_id = f"ORD-{START_INDEX + i:06d}"

        # ----------------------------------------------------
        # Item category
        # ----------------------------------------------------

        category = rng.choice(
            ITEM_CATEGORIES,
            p=[0.28, 0.22, 0.20, 0.15, 0.15]
        )

        # ----------------------------------------------------
        # Transaction amount
        # ----------------------------------------------------

        amount = float(
            np.round(
                rng.lognormal(
                    mean=6.5,
                    sigma=0.9
                ),
                2
            )
        )

        # Keep incoming amounts within demo range
        amount = min(
            max(amount, 100),
            15000
        )

        # ----------------------------------------------------
        # Reason code
        # ----------------------------------------------------

        reason_code = rng.choice(
            REASON_CODES,
            p=[0.30, 0.30, 0.25, 0.15]
        )

        # ----------------------------------------------------
        # Dispute timing
        # ----------------------------------------------------

        days_to_dispute = int(
            rng.integers(
                MIN_DAYS_TO_DISPUTE,
                MAX_DAYS_TO_DISPUTE + 1
            )
        )

        # ----------------------------------------------------
        # Evidence / case facts
        # ----------------------------------------------------

        delivery_confirmed = bool(
            rng.random() < 0.80
        )

        otp_auth_confirmed = bool(
            rng.random() < 0.85
        )

        shipping_billing_match = bool(
            rng.random() < 0.88
        )

        prior_complaint_on_file = bool(
            rng.random() < 0.10
        )

        # ----------------------------------------------------
        # Construct incoming dispute
        # ----------------------------------------------------

        rows.append(
            {
                "dispute_id": dispute_id,
                "order_id": order_id,
                "reason_code": reason_code,
                "amount": amount,
                "item_category": category,
                "delivery_confirmed": delivery_confirmed,
                "otp_auth_confirmed": otp_auth_confirmed,
                "shipping_billing_match": shipping_billing_match,
                "prior_complaint_on_file": prior_complaint_on_file,
                "days_to_dispute": days_to_dispute,
            }
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    pool_df = pd.DataFrame(rows)

    # ========================================================
    # ENSURE DATA DIRECTORY EXISTS
    # ========================================================

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    pool_df.to_csv(
        OUT_PATH,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print("INCOMING DISPUTE POOL GENERATED")
    print("=" * 70)

    print(f"Rows: {len(pool_df)}")
    print(f"Output: {OUT_PATH}")

    print()
    print("Reason code distribution:")
    print(
        pool_df["reason_code"]
        .value_counts()
        .to_string()
    )

    print()
    print("ID range:")
    print(
        f"First dispute: {pool_df['dispute_id'].iloc[0]}"
    )
    print(
        f"Last dispute:  {pool_df['dispute_id'].iloc[-1]}"
    )

    print()
    print("CSV columns:")
    print(
        pool_df.columns.tolist()
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_pool()