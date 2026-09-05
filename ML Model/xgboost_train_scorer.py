"""
Train the evidence-strength (win-probability) scorer.

IMPORTANT — feature discipline:
  - The model ONLY sees observable order/dispute features.
  - true_win_probability (hidden ground truth formula output) is NEVER
    a feature — only outcome_true is used, and only as the training label.
  - Metrics are reported on the held-out test.csv split, generated once
    by split_data.py and never touched during training/tuning.

Run:
    python train_scorer.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

DATA_DIR = "data"
MODEL_DIR = "models"
SEED = 42

# ---------------------------------------------------------------------------
# Feature columns — observable only. No leakage from golden_answers.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "amount",
    "days_to_dispute",
    "prior_dispute_count_at_time",
]

BOOLEAN_FEATURES = [
    "delivery_confirmed",
    "otp_auth_confirmed",
    "shipping_billing_match",
    "prior_complaint_on_file",
]

CATEGORICAL_FEATURES = [
    "reason_code",
    "item_category",
]

LABEL_COL = "outcome_true"


def load_split(name):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    return df


def build_features(df, encoders=None, fit=False):
    """One-hot encode categoricals; cast booleans to int."""
    X = df[NUMERIC_FEATURES].copy()

    for col in BOOLEAN_FEATURES:
        X[col] = df[col].astype(int)

    cat_dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    X = pd.concat([X, cat_dummies], axis=1)

    if fit:
        feature_columns = X.columns.tolist()
        return X, feature_columns
    else:
        # align to training columns, fill missing dummy cols with 0
        for col in encoders:
            if col not in X.columns:
                X[col] = 0
        X = X[encoders]
        return X


def main():
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    y_train = train_df[LABEL_COL].astype(int)
    y_val = val_df[LABEL_COL].astype(int)
    y_test = test_df[LABEL_COL].astype(int)

    X_train, feature_columns = build_features(train_df, fit=True)
    X_val = build_features(val_df, encoders=feature_columns)
    X_test = build_features(test_df, encoders=feature_columns)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Features ({len(feature_columns)}): {feature_columns}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=SEED,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # -------------------------------------------------------------------
    # Threshold selection on VALIDATION set only (never touch test set
    # until final reporting — avoids leaking test info into tuning)
    #
    # Cost-optimal threshold: minimizes total expected Rs. cost, not F1.
    #   FP (predict FIGHT, actually LOSE) -> cost = FIGHT_COST_FLAT
    #   FN (predict CONCEDE, actually WIN) -> cost = amount (money left on table)
    # Swept over a fine grid so the result isn't an artifact of a sparse
    # probability grid (see threshold_comparison.py for the full comparison
    # against F1-optimal and Youden's J).
    # -------------------------------------------------------------------
    FIGHT_COST_FLAT = 300.0

    val_probs = model.predict_proba(X_val)[:, 1]
    val_amounts = val_df["amount"].values

    candidate_thresholds = np.linspace(0.0, 1.0, 1000)
    best_threshold = 0.5
    best_cost = np.inf
    for t in candidate_thresholds:
        preds = (val_probs >= t).astype(int)
        fp_mask = (preds == 1) & (y_val.values == 0)
        fn_mask = (preds == 0) & (y_val.values == 1)
        total_cost = fp_mask.sum() * FIGHT_COST_FLAT + val_amounts[fn_mask].sum()
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = float(t)

    print(f"\nSelected threshold (val cost-optimal, fight_cost=Rs.{FIGHT_COST_FLAT:.0f}): {best_threshold:.3f}")
    print(f"Validation total cost at this threshold: Rs.{best_cost:,.2f}")

    # -------------------------------------------------------------------
    # FINAL evaluation on held-out TEST set — reported honestly
    # -------------------------------------------------------------------
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    print("\n=== HELD-OUT TEST SET METRICS ===")
    print(classification_report(y_test, test_preds, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_preds))
    print(f"ROC-AUC : {roc_auc_score(y_test, test_probs):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, test_probs):.4f}")

    # -------------------------------------------------------------------
    # SHAP feature importance (global, for explainability layer later)
    # -------------------------------------------------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_columns, mean_abs_shap), key=lambda x: -x[1]
    )
    print("\nTop feature importances (mean |SHAP|):")
    for name, val in importance[:10]:
        print(f"  {name:<35} {val:.4f}")

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, f"{MODEL_DIR}/xgb_scorer.joblib")

    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(f"{MODEL_DIR}/threshold.json", "w") as f:
        json.dump({"threshold": best_threshold, "strategy": "cost_optimal", "fight_cost_flat": FIGHT_COST_FLAT}, f, indent=2)

    metrics_summary = {
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
        "test_pr_auc": float(average_precision_score(y_test, test_probs)),
        "threshold": best_threshold,
        "threshold_strategy": "cost_optimal",
        "fight_cost_flat": FIGHT_COST_FLAT,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    with open(f"{MODEL_DIR}/metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved model + artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()