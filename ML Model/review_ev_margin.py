"""
Sensitivity sweep: how does REVIEW_MARGIN trade off:
  - % of disputes auto-conceded that were actually winnable
    (money left on table)
  - % of disputes routed to manual review (human workload)
  - total modeled cost under each margin

IMPORTANT:
  - Uses the VALIDATION set, NOT the held-out TEST set.
  - The validation set is used to choose the REVIEW_MARGIN.
  - The TEST set remains untouched for final evaluation.
  - The trained model and FIGHT_COST remain fixed.
  - Only REVIEW_MARGIN is varied.

This follows the same evaluation discipline as threshold selection:
  TRAIN -> model training
  VALIDATION -> threshold + review-margin selection
  TEST -> final one-time evaluation

outcome_true is used ONLY for diagnostic cost reporting. It is never
fed into the decision logic.

Run:
    python review_margin_sweep.py
"""

import json

import joblib
import numpy as np
import pandas as pd


DATA_DIR = "data"
MODEL_DIR = "models"

# Current project assumption:
# incremental operational/evidence-processing cost of fighting a dispute.
FIGHT_COST_FLAT = 300.0

# Candidate human-review safety margins to compare.
MARGIN_CANDIDATES = [0, 50, 100, 150, 200, 300, 400, 500]


def load_features(df, feature_columns):
    """
    Recreate exactly the feature representation used by the trained XGBoost
    scorer. No outcome/golden-answer columns are used as model features.
    """

    X = df[
        [
            "amount",
            "days_to_dispute",
            "prior_dispute_count_at_time",
        ]
    ].copy()

    # Boolean features
    for col in [
        "delivery_confirmed",
        "otp_auth_confirmed",
        "shipping_billing_match",
        "prior_complaint_on_file",
    ]:
        X[col] = df[col].astype(int)

    # Categorical features
    cat_dummies = pd.get_dummies(
        df[["reason_code", "item_category"]],
        prefix=["reason_code", "item_category"],
    )

    X = pd.concat([X, cat_dummies], axis=1)

    # Align exactly with the columns used during training.
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    return X[feature_columns]


def decide(win_probability, amount, fight_cost, review_margin):
    """
    Deterministic business decision layer.

    EV of fighting:
        EV = win_probability * amount - fight_cost

    Decision:
        EV > +review_margin -> FIGHT
        EV < -review_margin -> CONCEDE
        Otherwise           -> MANUAL_REVIEW
    """

    ev_fight = win_probability * amount - fight_cost

    if ev_fight > review_margin:
        return "FIGHT"

    elif ev_fight < -review_margin:
        return "CONCEDE"

    else:
        return "MANUAL_REVIEW"


def main():

    # ---------------------------------------------------------
    # Load trained scorer
    # ---------------------------------------------------------

    model = joblib.load(
        f"{MODEL_DIR}/xgb_scorer.joblib"
    )

    with open(
        f"{MODEL_DIR}/feature_columns.json"
    ) as f:
        feature_columns = json.load(f)

    # ---------------------------------------------------------
    # VALIDATION SET ONLY
    # ---------------------------------------------------------

    df = pd.read_csv(
        f"{DATA_DIR}/val.csv"
    )

    X = load_features(
        df,
        feature_columns
    )

    # Model prediction only.
    # outcome_true is NOT used here.
    df["win_probability"] = model.predict_proba(X)[:, 1]

    total_disputes = len(df)

    # Used only for diagnostic context.
    total_winnable_amount = df.loc[
        df["outcome_true"] == True,
        "amount"
    ].sum()

    print("=" * 90)
    print("REVIEW MARGIN SENSITIVITY — VALIDATION SET")
    print("=" * 90)

    print(
        f"\nFight cost assumption: ₹{FIGHT_COST_FLAT:,.0f}"
    )

    print(
        "\n"
        f"{'Margin':>8} "
        f"{'FIGHT%':>8} "
        f"{'CONCEDE%':>10} "
        f"{'REVIEW%':>9} "
        f"{'MoneyLeftOnTable':>20} "
        f"{'FightLossCost':>16} "
        f"{'TotalCost':>16}"
    )

    results = []

    # ---------------------------------------------------------
    # Sweep review margins
    # ---------------------------------------------------------

    for margin in MARGIN_CANDIDATES:

        decisions = [
            decide(
                probability,
                amount,
                FIGHT_COST_FLAT,
                margin,
            )
            for probability, amount
            in zip(
                df["win_probability"],
                df["amount"],
            )
        ]

        decisions = pd.Series(
            decisions,
            index=df.index,
        )

        fight_mask = decisions == "FIGHT"
        concede_mask = decisions == "CONCEDE"
        review_mask = decisions == "MANUAL_REVIEW"

        # -----------------------------------------------------
        # Diagnostic only:
        # money left on table
        # -----------------------------------------------------

        money_left = df.loc[
            concede_mask
            & (df["outcome_true"] == True),
            "amount",
        ].sum()

        # -----------------------------------------------------
        # Diagnostic only:
        # fight-and-lose operational cost
        # -----------------------------------------------------

        n_fight_lost = (
            fight_mask
            & (df["outcome_true"] == False)
        ).sum()

        fight_loss_cost = (
            n_fight_lost
            * FIGHT_COST_FLAT
        )

        # Total modeled cost
        total_cost = (
            money_left
            + fight_loss_cost
        )

        # Routing percentages
        pct_fight = (
            fight_mask.mean()
            * 100
        )

        pct_concede = (
            concede_mask.mean()
            * 100
        )

        pct_review = (
            review_mask.mean()
            * 100
        )

        print(
            f"{margin:>8} "
            f"{pct_fight:>7.1f}% "
            f"{pct_concede:>9.1f}% "
            f"{pct_review:>8.1f}% "
            f"₹{money_left:>17,.0f} "
            f"₹{fight_loss_cost:>13,.0f} "
            f"₹{total_cost:>13,.0f}"
        )

        results.append(
            {
                "review_margin": margin,
                "pct_fight": float(pct_fight),
                "pct_concede": float(pct_concede),
                "pct_manual_review": float(pct_review),
                "money_left_on_table": float(money_left),
                "fight_loss_cost": float(fight_loss_cost),
                "total_cost": float(total_cost),
            }
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("VALIDATION SET SUMMARY")
    print("=" * 90)

    print(
        f"Total disputes: "
        f"{total_disputes:,}"
    )

    print(
        f"Total winnable dispute value: "
        f"₹{total_winnable_amount:,.0f}"
    )

    # Find minimum modeled cost.
    best_result = min(
        results,
        key=lambda x: x["total_cost"]
    )

    print(
        "\nLowest modeled validation cost:"
    )

    print(
        f"  REVIEW_MARGIN = "
        f"₹{best_result['review_margin']:,.0f}"
    )

    print(
        f"  Total cost = "
        f"₹{best_result['total_cost']:,.0f}"
    )

    print(
        f"  FIGHT = "
        f"{best_result['pct_fight']:.1f}%"
    )

    print(
        f"  CONCEDE = "
        f"{best_result['pct_concede']:.1f}%"
    )

    print(
        f"  MANUAL_REVIEW = "
        f"{best_result['pct_manual_review']:.1f}%"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output = {
        "dataset": "validation",
        "fight_cost_flat": FIGHT_COST_FLAT,
        "margin_candidates": MARGIN_CANDIDATES,
        "results": results,
        "lowest_validation_cost_margin": best_result[
            "review_margin"
        ],
    }

    with open(
        f"{MODEL_DIR}/review_margin_sweep.json",
        "w",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\nSaved sweep to "
        f"{MODEL_DIR}/review_margin_sweep.json"
    )


if __name__ == "__main__":
    main()