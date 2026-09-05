"""
Split disputes into train / validation / held-out test sets for the ML scorer.

Rules:
  - Excludes 'is_unknown_reason_edge_case' rows entirely from ML splits.
    Those exist only for the RAG/graceful-failure demo path, not for scoring.
  - Stratifies by (reason_code, outcome_true) so class balance is consistent
    across splits.
  - Joins orders + disputes into one feature table; outcome_true/
    true_win_probability come from golden_answers.csv but are kept in a
    SEPARATE column set clearly marked as label/eval-only, never fed as a
    predictive feature.

Run:
    python split_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SEED = 42

orders = pd.read_csv(f"{DATA_DIR}/orders.csv")
disputes = pd.read_csv(f"{DATA_DIR}/disputes.csv")
golden = pd.read_csv(f"{DATA_DIR}/golden_answers.csv")

# ---------------------------------------------------------------------------
# Join everything on dispute_id / order_id
# ---------------------------------------------------------------------------
df = disputes.merge(golden, on="dispute_id", how="left")
df = df.merge(orders, on="order_id", how="left")

# ---------------------------------------------------------------------------
# Exclude unknown-reason edge cases from ML training entirely
# ---------------------------------------------------------------------------
ml_df = df[~df["is_unknown_reason_edge_case"]].copy()
excluded_df = df[df["is_unknown_reason_edge_case"]].copy()

print(f"Total disputes           : {len(df)}")
print(f"Eligible for ML splits   : {len(ml_df)}")
print(f"Excluded (unknown reason): {len(excluded_df)}")

# ---------------------------------------------------------------------------
# Stratify key: combine reason_code + outcome_true so both are balanced
# ---------------------------------------------------------------------------
ml_df["strata"] = ml_df["reason_code"] + "_" + ml_df["outcome_true"].astype(str)

train_df, temp_df = train_test_split(
    ml_df,
    test_size=0.30,
    random_state=SEED,
    stratify=ml_df["strata"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_df["strata"],
)

for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    split = split.drop(columns=["strata"])
    split.to_csv(f"{DATA_DIR}/{name}.csv", index=False)
    print(f"\n{name}.csv: {len(split)} rows")
    print(split.groupby("reason_code")["outcome_true"].mean().round(3))

# Held-out unknown-reason cases saved separately for the RAG/fallback demo
excluded_df.to_csv(f"{DATA_DIR}/unknown_reason_cases.csv", index=False)
print(f"\nunknown_reason_cases.csv: {len(excluded_df)} rows (fallback demo only)")