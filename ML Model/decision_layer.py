"""
Decision + Priority layer.

Takes the scorer's win_probability output and combines it with dispute
amount and urgency to produce:
  - A routing decision: FIGHT / CONCEDE / MANUAL_REVIEW
  - A priority score, so the dashboard can surface high-value/urgent cases
    first in the manual-review queue.

This layer does NOT retrain anything - it's pure decision logic on top of
the already-trained scorer's probability output. Kept deliberately simple
(explicit formula, no black box) since every money decision must be
explainable per the track's requirements.

Decision rule:
  expected_value_fight = win_probability * amount - FIGHT_COST_FLAT
  expected_value_concede = 0   (baseline: money already assumed lost)

  if expected_value_fight - expected_value_concede > REVIEW_MARGIN:
      -> FIGHT (auto)
  elif expected_value_fight - expected_value_concede < -REVIEW_MARGIN:
      -> CONCEDE (auto)
  else:
      -> MANUAL_REVIEW (ambiguous zone, human decides)

REVIEW_MARGIN creates the human-in-the-loop band: cases where the EV
calculation is too close to call get routed to a person instead of an
automatic decision. This directly satisfies "every money action
explainable, bounded and gated."

Priority score (for manual-review queue ordering):
  priority = amount_weight * normalized_amount + urgency_weight * urgency
  where urgency = 1 - (days_left / max_response_window)
  -> higher amount and closer deadline = higher priority

Run:
    python decision_layer.py
"""

import json

import joblib
import numpy as np
import pandas as pd

DATA_DIR = "data"
MODEL_DIR = "models"

FIGHT_COST_FLAT = 300.0        # same assumption as threshold_comparison.py
REVIEW_MARGIN = 150.0          # EV band width (Rs.) that triggers manual review
MAX_RESPONSE_WINDOW_DAYS = 20  # typical dispute response deadline

AMOUNT_WEIGHT = 0.6
URGENCY_WEIGHT = 0.4


def load_features(df, feature_columns):
    X = df[["amount", "days_to_dispute", "prior_dispute_count_at_time"]].copy()
    for col in ["delivery_confirmed", "otp_auth_confirmed", "shipping_billing_match", "prior_complaint_on_file"]:
        X[col] = df[col].astype(int)
    cat_dummies = pd.get_dummies(df[["reason_code", "item_category"]], prefix=["reason_code", "item_category"])
    X = pd.concat([X, cat_dummies], axis=1)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    return X[feature_columns]


def decide(win_probability, amount, fight_cost=FIGHT_COST_FLAT, review_margin=REVIEW_MARGIN):
    ev_fight = win_probability * amount - fight_cost
    ev_concede = 0.0
    ev_diff = ev_fight - ev_concede

    if ev_diff > review_margin:
        decision = "FIGHT"
    elif ev_diff < -review_margin:
        decision = "CONCEDE"
    else:
        decision = "MANUAL_REVIEW"

    return decision, ev_fight, ev_diff


def compute_priority(amount, days_left, all_amounts):
    # normalize amount against the batch's amount range (0-1)
    amount_norm = (amount - all_amounts.min()) / (all_amounts.max() - all_amounts.min() + 1e-9)
    urgency = 1 - np.clip(days_left / MAX_RESPONSE_WINDOW_DAYS, 0, 1)
    priority = AMOUNT_WEIGHT * amount_norm + URGENCY_WEIGHT * urgency
    return float(priority)


def main():
    model = joblib.load(f"{MODEL_DIR}/xgb_scorer.joblib")
    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)

    # Run the full decision layer over the held-out test set as a demo batch
    df = pd.read_csv(f"{DATA_DIR}/test.csv")
    X = load_features(df, feature_columns)
    df["win_probability"] = model.predict_proba(X)[:, 1]

    # days_left = response window remaining (assume dispute just arrived,
    # so days_left = MAX_RESPONSE_WINDOW_DAYS - days_to_dispute as a proxy
    # for how much of the response clock has already elapsed)
    df["days_left"] = (MAX_RESPONSE_WINDOW_DAYS - df["days_to_dispute"]).clip(lower=0)

    decisions, ev_fights, ev_diffs = [], [], []
    for _, row in df.iterrows():
        d, ev_f, ev_d = decide(row["win_probability"], row["amount"])
        decisions.append(d)
        ev_fights.append(ev_f)
        ev_diffs.append(ev_d)

    df["decision"] = decisions
    df["ev_fight"] = ev_fights
    df["ev_diff"] = ev_diffs

    all_amounts = df["amount"]
    df["priority_score"] = df.apply(
        lambda r: compute_priority(r["amount"], r["days_left"], all_amounts), axis=1
    )

    # ---- Summary ----
    print("Decision distribution:")
    print(df["decision"].value_counts())
    print(f"\n% routed to manual review: {(df['decision']=='MANUAL_REVIEW').mean()*100:.1f}%")

    # sanity: for FIGHT decisions, what fraction actually won (using outcome_true
    # if present via golden join - only for this diagnostic print, not used
    # anywhere in the decision logic itself)
    if "outcome_true" in df.columns:
        fight_df = df[df["decision"] == "FIGHT"]
        concede_df = df[df["decision"] == "CONCEDE"]
        print(f"\nOf {len(fight_df)} FIGHT decisions: {fight_df['outcome_true'].mean()*100:.1f}% actually won")
        print(f"Of {len(concede_df)} CONCEDE decisions: {concede_df['outcome_true'].mean()*100:.1f}% would have won (money left on table)")

    print("\nTop 10 highest-priority MANUAL_REVIEW cases:")
    review_queue = df[df["decision"] == "MANUAL_REVIEW"].sort_values("priority_score", ascending=False)
    cols = ["dispute_id", "reason_code", "amount", "win_probability", "days_left", "priority_score"]
    print(review_queue[cols].head(10).to_string(index=False))

    df.to_csv(f"{DATA_DIR}/decision_layer_output_demo.csv", index=False)
    print(f"\nSaved full decision output to {DATA_DIR}/decision_layer_output_demo.csv")


if __name__ == "__main__":
    main()