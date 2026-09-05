"""
Threshold strategy comparison for the evidence-strength scorer.

Compares three thresholding strategies, ALL computed on the VALIDATION set:
  1. F1-optimal        - maximizes harmonic mean of precision/recall
  2. Youden's J         - maximizes (sensitivity + specificity - 1)
  3. Cost-optimal       - minimizes total expected $ cost of wrong decisions

The chosen threshold is then applied EXACTLY ONCE to the held-out test set
for final reporting. Test set is never used to pick a threshold.

Cost assumptions (edit these to match your pitch's stated numbers):
  - False Positive (predict WIN, actually LOSE): merchant fights and loses.
    Cost = cost_of_fighting (dispute-fighting fee/effort) + the dispute
    amount is lost anyway (same as if they'd conceded) -> the INCREMENTAL
    cost of fighting-and-losing vs conceding immediately is just the
    fighting cost itself.
  - False Negative (predict LOSE, actually WIN): merchant concedes a
    winnable dispute. Cost = the full dispute amount, since they gave up
    money they could have recovered.
  - True Positive / True Negative: no incremental cost beyond baseline.

This asymmetry (FN cost >> FP cost, and FN cost scales with amount) is
the whole reason a cost-optimal threshold differs meaningfully from F1
or Youden's J here.

Run:
    python threshold_comparison.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

DATA_DIR = "data"
MODEL_DIR = "models"

# Flat cost of attempting to fight a dispute (documentation/ops effort),
# independent of amount. Adjust to whatever you state in your pitch.
FIGHT_COST_FLAT = 300.0


def load_features_and_labels(split_name, feature_columns):
    df = pd.read_csv(f"{DATA_DIR}/{split_name}.csv")

    X = df[["amount", "days_to_dispute", "prior_dispute_count_at_time"]].copy()
    for col in ["delivery_confirmed", "otp_auth_confirmed", "shipping_billing_match", "prior_complaint_on_file"]:
        X[col] = df[col].astype(int)
    cat_dummies = pd.get_dummies(df[["reason_code", "item_category"]], prefix=["reason_code", "item_category"])
    X = pd.concat([X, cat_dummies], axis=1)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    y = df["outcome_true"].astype(int)
    amounts = df["amount"].values
    return df, X, y, amounts


def f1_optimal_threshold(y_true, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def youden_j_threshold(y_true, probs):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def cost_optimal_threshold(y_true, probs, amounts, fight_cost_flat=FIGHT_COST_FLAT):
    """
    Sweep candidate thresholds; for each, compute total expected cost:
      - FP (predict fight, actually lose): cost = fight_cost_flat
      - FN (predict concede, actually win): cost = amount (money left on table)
    Pick threshold minimizing total cost.

    Uses the SAME fine-grained grid (np.linspace(0,1,1000)) that F1/Youden
    are effectively drawn from, so the comparison is apples-to-apples and
    cost-optimal is guaranteed to find a threshold whose cost is <= the
    cost of the F1-optimal or Youden's J thresholds evaluated on this grid.
    """
    candidate_thresholds = np.linspace(0.0, 1.0, 1000)

    best_threshold = 0.5
    best_cost = np.inf
    cost_curve = []

    for t in candidate_thresholds:
        preds = (probs >= t).astype(int)
        fp_mask = (preds == 1) & (y_true == 0)
        fn_mask = (preds == 0) & (y_true == 1)

        fp_cost = fp_mask.sum() * fight_cost_flat
        fn_cost = amounts[fn_mask].sum()
        total_cost = fp_cost + fn_cost

        cost_curve.append((float(t), float(total_cost)))

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(t)

    return best_threshold, best_cost, cost_curve


def evaluate_at_threshold(y_true, probs, threshold, amounts, label):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    fp_cost = fp * FIGHT_COST_FLAT
    fn_cost = amounts[(preds == 0) & (y_true == 1)].sum()
    total_cost = fp_cost + fn_cost

    print(f"\n--- {label} (threshold={threshold:.3f}) ---")
    print(classification_report(y_true, preds, digits=3))
    print(f"Confusion matrix:\n{cm}")
    print(f"FP cost (fought & lost, {fp} cases x Rs.{FIGHT_COST_FLAT}) = Rs.{fp_cost:,.2f}")
    print(f"FN cost (conceded a winnable case, {fn} cases, sum of amounts) = Rs.{fn_cost:,.2f}")
    print(f"TOTAL EXPECTED COST = Rs.{total_cost:,.2f}")
    return {
        "threshold": threshold,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
        "total_cost": float(total_cost),
    }


def main():
    model = joblib.load(f"{MODEL_DIR}/xgb_scorer.joblib")
    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)

    # ---- VALIDATION: compute all three candidate thresholds ----
    val_df, X_val, y_val, amounts_val = load_features_and_labels("val", feature_columns)
    val_probs = model.predict_proba(X_val)[:, 1]

    f1_t, f1_score = f1_optimal_threshold(y_val, val_probs)
    yj_t, yj_score = youden_j_threshold(y_val, val_probs)
    cost_t, cost_val, cost_curve = cost_optimal_threshold(y_val, val_probs, amounts_val)

    print("=" * 70)
    print("THRESHOLD CANDIDATES (selected on VALIDATION set only)")
    print("=" * 70)
    print(f"F1-optimal threshold      : {f1_t:.3f}  (F1={f1_score:.3f})")
    print(f"Youden's J threshold      : {yj_t:.3f}  (J={yj_score:.3f})")
    print(f"Cost-optimal threshold    : {cost_t:.3f}  (val total cost=Rs.{cost_val:,.2f})")

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE ON VALIDATION SET")
    print("=" * 70)
    results = {}
    results["f1"] = evaluate_at_threshold(y_val, val_probs, f1_t, amounts_val, "F1-optimal (val)")
    results["youden"] = evaluate_at_threshold(y_val, val_probs, yj_t, amounts_val, "Youden's J (val)")
    results["cost"] = evaluate_at_threshold(y_val, val_probs, cost_t, amounts_val, "Cost-optimal (val)")

    # ---- Choose cost-optimal as final strategy (per project priority) ----
    CHOSEN_STRATEGY = "cost"
    chosen_threshold = results[CHOSEN_STRATEGY]["threshold"]

    print("\n" + "=" * 70)
    print(f"CHOSEN STRATEGY: {CHOSEN_STRATEGY} -> threshold = {chosen_threshold:.3f}")
    print("Applying ONCE to held-out TEST set for final reporting")
    print("=" * 70)

    # ---- FINAL: apply once to held-out test ----
    test_df, X_test, y_test, amounts_test = load_features_and_labels("test", feature_columns)
    test_probs = model.predict_proba(X_test)[:, 1]

    final = evaluate_at_threshold(y_test, test_probs, chosen_threshold, amounts_test, "FINAL (held-out TEST)")
    print(f"\nTest ROC-AUC : {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Test PR-AUC  : {average_precision_score(y_test, test_probs):.4f}")

    # ---- Save everything ----
    output = {
        "candidates": {
            "f1_optimal": {"threshold": f1_t, "val_metrics": results["f1"]},
            "youden_j": {"threshold": yj_t, "val_metrics": results["youden"]},
            "cost_optimal": {"threshold": cost_t, "val_metrics": results["cost"]},
        },
        "chosen_strategy": CHOSEN_STRATEGY,
        "chosen_threshold": chosen_threshold,
        "final_test_metrics": final,
        "fight_cost_flat_assumption": FIGHT_COST_FLAT,
    }
    with open(f"{MODEL_DIR}/threshold_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    # overwrite the single threshold.json used by the decision layer
    with open(f"{MODEL_DIR}/threshold.json", "w") as f:
        json.dump({"threshold": chosen_threshold, "strategy": CHOSEN_STRATEGY}, f, indent=2)

    print(f"\nSaved comparison to {MODEL_DIR}/threshold_comparison.json")
    print(f"Updated {MODEL_DIR}/threshold.json with chosen threshold")


if __name__ == "__main__":
    main()