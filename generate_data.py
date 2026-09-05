"""
Synthetic data generator for the Chargeback Evidence Responder project.

Produces three files under data/:
  - orders.csv          (visible to pipeline)
  - disputes.csv       (visible to pipeline)
  - golden_answers.csv  (HIDDEN - eval only, never fed to model/pipeline)

Run:
    python generate_data.py
"""

import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from faker import Faker


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42

N_ORDERS = 20000
N_DISPUTES = 15000
N_CUSTOMERS = 6000

# Fixed cutoff date for reproducible historical data.
AS_OF_DATE = pd.Timestamp("2026-09-05")

MIN_PURCHASE_DAYS_BACK = 30
MAX_PURCHASE_DAYS_BACK = 182

MIN_DAYS_TO_DISPUTE = 1
MAX_DAYS_TO_DISPUTE = 24

# Deliberate unknown-reason edge cases for fallback evaluation.
N_UNKNOWN = 15

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


# ---------------------------------------------------------------------------
# Synthetic category effects
# ---------------------------------------------------------------------------
#
# These are synthetic assumptions used only to create learnable patterns.
# They are NOT claims about real-world chargeback behavior.
# ---------------------------------------------------------------------------

CATEGORY_ABUSE_WEIGHT = {
    "apparel": 1.3,
    "electronics": 1.2,
    "groceries": 0.5,
    "digital_goods": 1.1,
    "home_goods": 0.8,
}

CATEGORY_MEAN_WEIGHT = np.mean(
    list(CATEGORY_ABUSE_WEIGHT.values())
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)

OUT_DIR = "data"


# ---------------------------------------------------------------------------
# Step 1: Generate customers
# ---------------------------------------------------------------------------
#
# Each customer gets a hidden "dispute tendency".
#
# This is used ONLY during synthetic-data generation to make some customers
# more likely to have disputes than others.
#
# It is NOT exposed as an ML feature.
# ---------------------------------------------------------------------------

customer_ids = [
    f"CUST-{i:05d}"
    for i in range(N_CUSTOMERS)
]

# Mostly low, with a small tail of repeat/serial disputers.
customer_dispute_tendency = np.random.beta(
    1.5,
    8,
    size=N_CUSTOMERS
)

customer_tendency_map = dict(
    zip(
        customer_ids,
        customer_dispute_tendency
    )
)


# ---------------------------------------------------------------------------
# Step 2: Generate orders
# ---------------------------------------------------------------------------

def random_purchase_date():
    """
    Generate a historical purchase date.

    The minimum 30-day gap ensures that adding the maximum dispute window
    cannot produce a future dispute date relative to AS_OF_DATE.
    """

    days_back = np.random.randint(
        MIN_PURCHASE_DAYS_BACK,
        MAX_PURCHASE_DAYS_BACK + 1
    )

    return (
        AS_OF_DATE
        - timedelta(days=int(days_back))
    )


orders = []

for i in range(N_ORDERS):

    order_id = f"ORD-{i:05d}"

    customer_id = random.choice(
        customer_ids
    )

    category = np.random.choice(
        ITEM_CATEGORIES,
        p=[
            0.28,
            0.22,
            0.20,
            0.15,
            0.15,
        ]
    )

    amount = float(
        np.round(
            np.random.lognormal(
                mean=6.5,
                sigma=0.9
            ),
            2
        )
    )

    amount = min(
        max(amount, 100),
        15000
    )

    purchase_date = random_purchase_date()

    delivery_confirmed = (
        np.random.rand() < 0.80
    )

    delivery_date = (
        purchase_date
        + timedelta(
            days=int(
                np.random.randint(1, 10)
            )
        )
        if delivery_confirmed
        else pd.NaT
    )

    otp_auth_confirmed = (
        np.random.rand() < 0.85
    )

    shipping_billing_match = (
        np.random.rand() < 0.88
    )

    prior_complaint_on_file = (
        np.random.rand() < 0.10
    )

    orders.append(
        {
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "item_category": category,
            "purchase_date": (
                purchase_date
                .date()
                .isoformat()
            ),
            "delivery_confirmed": delivery_confirmed,
            "delivery_date": (
                delivery_date
                .date()
                .isoformat()
                if pd.notna(delivery_date)
                else ""
            ),
            "otp_auth_confirmed": otp_auth_confirmed,
            "shipping_billing_match": (
                shipping_billing_match
            ),
            "prior_complaint_on_file": (
                prior_complaint_on_file
            ),
        }
    )


orders_df = pd.DataFrame(orders)


# ---------------------------------------------------------------------------
# Step 3: Select orders that will receive disputes
# ---------------------------------------------------------------------------
#
# One order -> at most one dispute.
#
# Customers with higher hidden dispute tendencies are more likely to have
# an order selected for dispute generation.
#
# customer_dispute_tendency is NOT an ML feature.
# ---------------------------------------------------------------------------

order_customer_tendency = (
    orders_df["customer_id"]
    .map(customer_tendency_map)
    .to_numpy()
)

# Small non-zero probability for every order.
selection_weights = (
    0.05
    + order_customer_tendency
)

selection_probabilities = (
    selection_weights
    / selection_weights.sum()
)

dispute_order_idx = np.random.choice(
    orders_df.index,
    size=N_DISPUTES,
    replace=False,
    p=selection_probabilities
)

order_rows = (
    orders_df
    .loc[dispute_order_idx]
    .copy()
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------------
# Step 4: Generate disputes
# ---------------------------------------------------------------------------

# Track customer's running prior dispute count.
cust_dispute_count = {
    customer_id: 0
    for customer_id in customer_ids
}

disputes = []
golden = []

# Rulebook chunk IDs.
# These MUST match generate_rulebook.py.
CHUNK_ID_MAP = {
    code: f"RULE_{code.upper()}"
    for code in REASON_CODES
}


# ---------------------------------------------------------------------------
# Generate dispute attributes
# ---------------------------------------------------------------------------

for _, row in order_rows.iterrows():

    order_id = row["order_id"]

    customer_id = row["customer_id"]

    reason_code = np.random.choice(
        REASON_CODES,
        p=[
            0.30,
            0.30,
            0.25,
            0.15,
        ]
    )

    purchase_date = pd.Timestamp(
        row["purchase_date"]
    )

    days_to_dispute = int(
        np.random.randint(
            MIN_DAYS_TO_DISPUTE,
            MAX_DAYS_TO_DISPUTE + 1
        )
    )

    dispute_date = (
        purchase_date
        + timedelta(
            days=days_to_dispute
        )
    )

    # Safety check: no future disputes.
    if dispute_date > AS_OF_DATE:
        raise ValueError(
            f"Generated future dispute date: "
            f"{dispute_date}"
        )


    # Customer's prior disputes at this point in time.
    prior_dispute_count = (
        cust_dispute_count[
            customer_id
        ]
    )

    cust_dispute_count[
        customer_id
    ] += 1


    # -----------------------------------------------------------------------
    # Observable features
    # -----------------------------------------------------------------------

    amount = float(
        row["amount"]
    )

    category = row[
        "item_category"
    ]

    delivery_confirmed = bool(
        row["delivery_confirmed"]
    )

    otp_auth_confirmed = bool(
        row["otp_auth_confirmed"]
    )

    shipping_billing_match = bool(
        row["shipping_billing_match"]
    )

    prior_complaint_on_file = bool(
        row["prior_complaint_on_file"]
    )


    # -----------------------------------------------------------------------
    # TRUE WIN PROBABILITY
    # -----------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This is hidden ground truth.
    #
    # The ML model will NOT receive this formula.
    #
    # The model only sees the observable features and learns their
    # relationship with the actual outcome.
    # -----------------------------------------------------------------------

    base = 0.05


    # -----------------------------------------------------------------------
    # Evidence signals
    # -----------------------------------------------------------------------

    base += (
        0.22
        if delivery_confirmed
        else 0.00
    )

    base += (
        0.18
        if otp_auth_confirmed
        else 0.00
    )

    base += (
        0.07
        if shipping_billing_match
        else 0.00
    )

    base += (
        0.07
        if not prior_complaint_on_file
        else 0.00
    )

    base += (
        0.06
        if days_to_dispute < 7
        else 0.00
    )


    # -----------------------------------------------------------------------
    # Repeat-disputer effect
    # -----------------------------------------------------------------------

    base -= (
        0.05
        * min(
            prior_dispute_count,
            4
        )
    )


    # -----------------------------------------------------------------------
    # AMOUNT EFFECT
    # -----------------------------------------------------------------------
    #
    # Amount is a moderate ML signal.
    #
    # It is also used independently by the later priority layer.
    # -----------------------------------------------------------------------

    if amount >= 10000:

        base += 0.06

    elif amount >= 5000:

        base += 0.03

    elif amount < 1000:

        base -= 0.02


    # -----------------------------------------------------------------------
    # ITEM CATEGORY EFFECT
    # -----------------------------------------------------------------------

    category_weight = (
        CATEGORY_ABUSE_WEIGHT[
            category
        ]
    )

    category_effect = (
        category_weight
        - CATEGORY_MEAN_WEIGHT
    ) * 0.06

    base += category_effect


    # -----------------------------------------------------------------------
    # Reason-code-specific evidence effects
    # -----------------------------------------------------------------------

    if reason_code == "unauthorized_transaction":

        base += (
            0.12
            if otp_auth_confirmed
            else -0.10
        )


    elif reason_code == "item_not_received":

        base += (
            0.12
            if delivery_confirmed
            else -0.15
        )


    elif reason_code == "not_as_described":

        base += (
            0.04
            if not prior_complaint_on_file
            else -0.05
        )


    elif reason_code == "duplicate_charge":

        base += (
            0.08
            if shipping_billing_match
            else 0.00
        )

        base -= (
            0.04
            * min(
                prior_dispute_count,
                4
            )
        )


    # -----------------------------------------------------------------------
    # Noise
    # -----------------------------------------------------------------------
    #
    # Prevents the synthetic problem from becoming perfectly deterministic.
    # -----------------------------------------------------------------------

    noise = np.random.normal(
        0,
        0.06
    )

    true_win_prob = float(
        np.clip(
            base + noise,
            0.02,
            0.98
        )
    )


    # -----------------------------------------------------------------------
    # Actual hidden outcome
    # -----------------------------------------------------------------------

    outcome_true = bool(
        np.random.rand()
        < true_win_prob
    )


    # -----------------------------------------------------------------------
    # Expected agent trajectory
    # -----------------------------------------------------------------------

    if (
        reason_code == "item_not_received"
        and not delivery_confirmed
    ):

        expected_trajectory = (
            "retry_needed"
        )

    else:

        expected_trajectory = (
            "clean_fight"
            if true_win_prob > 0.5
            else "clean_concede"
        )


    # -----------------------------------------------------------------------
    # Visible dispute record
    # -----------------------------------------------------------------------

    disputes.append(
        {
            "dispute_id": None,
            "order_id": order_id,
            "reason_code": reason_code,
            "dispute_date": (
                dispute_date
                .date()
                .isoformat()
            ),
            "days_to_dispute": (
                days_to_dispute
            ),
            "status": "pending",
        }
    )


    # -----------------------------------------------------------------------
    # Hidden golden answer
    # -----------------------------------------------------------------------

    golden.append(
        {
            "dispute_id": None,

            "true_win_probability": round(
                true_win_prob,
                4
            ),

            "outcome_true": outcome_true,

            "correct_relevant_chunk_id": (
                CHUNK_ID_MAP[
                    reason_code
                ]
            ),

            "expected_trajectory": (
                expected_trajectory
            ),

            "prior_dispute_count_at_time": (
                prior_dispute_count
            ),

            "is_unknown_reason_edge_case": (
                False
            ),
        }
    )


# ---------------------------------------------------------------------------
# Step 5: Convert to DataFrames and sort chronologically
# ---------------------------------------------------------------------------

disputes_df = pd.DataFrame(
    disputes
)

golden_df = pd.DataFrame(
    golden
)

disputes_df[
    "dispute_date"
] = pd.to_datetime(
    disputes_df[
        "dispute_date"
    ]
)

sort_idx = (
    disputes_df[
        "dispute_date"
    ]
    .sort_values()
    .index
)

disputes_df = (
    disputes_df
    .loc[sort_idx]
    .reset_index(drop=True)
)

golden_df = (
    golden_df
    .loc[sort_idx]
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------------
# Assign stable dispute IDs after sorting
# ---------------------------------------------------------------------------

for i in range(
    len(disputes_df)
):

    dispute_id = (
        f"DSP-{i:05d}"
    )

    disputes_df.loc[
        i,
        "dispute_id"
    ] = dispute_id

    golden_df.loc[
        i,
        "dispute_id"
    ] = dispute_id


disputes_df[
    "dispute_date"
] = (
    pd.to_datetime(
        disputes_df[
            "dispute_date"
        ]
    )
    .dt.date
    .astype(str)
)


# ---------------------------------------------------------------------------
# Step 6: Inject deliberate unknown-reason edge cases
# ---------------------------------------------------------------------------
#
# These cases remain in disputes.csv for fallback testing.
#
# They are NOT normal ML training examples.
#
# Expected path:
#
#     unknown reason
#          ↓
#     no matching RAG rule
#          ↓
#     safe fallback
#          ↓
#     manual review
# ---------------------------------------------------------------------------

unknown_idx = np.random.choice(
    disputes_df.index,
    size=N_UNKNOWN,
    replace=False
)

disputes_df.loc[
    unknown_idx,
    "reason_code"
] = "other_unclassified"

golden_df.loc[
    unknown_idx,
    "correct_relevant_chunk_id"
] = "NONE"

golden_df.loc[
    unknown_idx,
    "expected_trajectory"
] = (
    "manual_review_fallback"
)

golden_df.loc[
    unknown_idx,
    "is_unknown_reason_edge_case"
] = True


# ---------------------------------------------------------------------------
# Step 7: Save files
# ---------------------------------------------------------------------------

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

orders_df.to_csv(
    f"{OUT_DIR}/orders.csv",
    index=False
)

disputes_df.to_csv(
    f"{OUT_DIR}/disputes.csv",
    index=False
)

golden_df.to_csv(
    f"{OUT_DIR}/golden_answers.csv",
    index=False
)


# ---------------------------------------------------------------------------
# Step 8: Validation / summary
# ---------------------------------------------------------------------------

print(
    f"orders.csv         : "
    f"{len(orders_df)} rows"
)

print(
    f"disputes.csv       : "
    f"{len(disputes_df)} rows"
)

print(
    f"golden_answers.csv : "
    f"{len(golden_df)} rows "
    "(HIDDEN - eval only)"
)


print(
    "\nReason code distribution:"
)

print(
    disputes_df[
        "reason_code"
    ].value_counts()
)


print(
    "\nTrue outcome balance:"
)

print(
    golden_df[
        "outcome_true"
    ].value_counts(
        normalize=True
    )
)


print(
    "\nExpected trajectory distribution:"
)

print(
    golden_df[
        "expected_trajectory"
    ].value_counts()
)


print(
    "\nUnknown edge cases:"
)

print(
    golden_df[
        "is_unknown_reason_edge_case"
    ].sum()
)


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------

print(
    "\nDate validation:"
)

purchase_dates = pd.to_datetime(
    orders_df[
        "purchase_date"
    ]
)

dispute_dates = pd.to_datetime(
    disputes_df[
        "dispute_date"
    ]
)

print(
    f"Latest purchase date : "
    f"{purchase_dates.max().date()}"
)

print(
    f"Latest dispute date  : "
    f"{dispute_dates.max().date()}"
)

print(
    f"AS_OF_DATE           : "
    f"{AS_OF_DATE.date()}"
)

print(
    "Future disputes      :",
    int(
        (
            dispute_dates
            > AS_OF_DATE
        ).sum()
    )
)


# ---------------------------------------------------------------------------
# Order uniqueness validation
# ---------------------------------------------------------------------------

print(
    "\nOrder uniqueness validation:"
)

print(
    "Unique disputed orders:",
    disputes_df[
        "order_id"
    ].nunique()
)

print(
    "Total disputes:",
    len(disputes_df)
)

print(
    "Duplicate disputed orders:",
    int(
        disputes_df[
            "order_id"
        ].duplicated().sum()
    )
)