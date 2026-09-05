"""
Generate the 4 core evaluation figures for the pitch/dashboard.

Figures produced (saved to figures/):
  1. shap_beeswarm.png        - SHAP summary plot, held-out test set
  2. review_margin_sensitivity.png - 2-panel: routing % vs margin, cost vs margin
  3. confusion_matrix.png     - held-out test set, at chosen cost-optimal threshold
  4. economic_error_cost.png  - fight-loss cost vs money-left-on-table, stacked bar

Run AFTER train_scorer.py and review_margin_sweep.py have both been run
(this script reads their saved artifacts, does not retrain anything).

Run:
    python make_figures.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import confusion_matrix

DATA_DIR = "data"
MODEL_DIR = "models"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# Consistent style across all 4 figures
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def build_features(df, feature_columns):
    X = df[["amount", "days_to_dispute", "prior_dispute_count_at_time"]].copy()
    for col in ["delivery_confirmed", "otp_auth_confirmed", "shipping_billing_match", "prior_complaint_on_file"]:
        X[col] = df[col].astype(int)
    cat_dummies = pd.get_dummies(df[["reason_code", "item_category"]], prefix=["reason_code", "item_category"])
    X = pd.concat([X, cat_dummies], axis=1)
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    return X[feature_columns]


def load_common():
    model = joblib.load(f"{MODEL_DIR}/xgb_scorer.joblib")
    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{MODEL_DIR}/threshold.json") as f:
        threshold_info = json.load(f)
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    X_test = build_features(test_df, feature_columns)
    return model, feature_columns, threshold_info, test_df, X_test


# ---------------------------------------------------------------------------
# 1. SHAP BEESWARM
# ---------------------------------------------------------------------------
def make_shap_beeswarm(model, X_test, feature_columns):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_columns,
        show=False,
        max_display=10,
    )
    plt.title("What drives the win-probability score?\n(SHAP values, held-out test set)")
    plt.tight_layout()
    path = f"{FIG_DIR}/shap_beeswarm.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 2. REVIEW MARGIN SENSITIVITY (2-panel)
# ---------------------------------------------------------------------------
def make_review_margin_sensitivity():
    with open(f"{MODEL_DIR}/review_margin_sweep.json") as f:
        sweep = json.load(f)

    results = sweep["results"] if isinstance(sweep, dict) and "results" in sweep else sweep
    margins = [r["review_margin"] for r in results]
    fight_pct = [r["pct_fight"] for r in results]
    concede_pct = [r["pct_concede"] for r in results]
    review_pct = [r["pct_manual_review"] for r in results]
    money_left = [r["money_left_on_table"] for r in results]
    fight_loss = [r["fight_loss_cost"] for r in results]
    total_cost = [r["total_cost"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: routing % vs margin
    ax = axes[0]
    ax.plot(margins, fight_pct, marker="o", label="FIGHT %", color="#2ca02c")
    ax.plot(margins, concede_pct, marker="o", label="CONCEDE %", color="#d62728")
    ax.plot(margins, review_pct, marker="o", label="MANUAL_REVIEW %", color="#1f77b4")
    ax.set_xlabel("Review Margin (Rs.)")
    ax.set_ylabel("% of disputes")
    ax.set_title("Routing vs Review Margin")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: cost vs margin
    ax = axes[1]
    ax.plot(margins, money_left, marker="o", label="Money Left on Table", color="#d62728")
    ax.plot(margins, fight_loss, marker="o", label="Fight-Loss Cost", color="#ff7f0e")
    ax.plot(margins, total_cost, marker="o", label="Total Modeled Cost", color="#1f77b4", linewidth=2.5)
    ax.set_xlabel("Review Margin (Rs.)")
    ax.set_ylabel("Rs.")
    ax.set_title("Modeled Cost vs Review Margin")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Review-Margin Sensitivity (Validation Set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = f"{FIG_DIR}/review_margin_sensitivity.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# 3. CONFUSION MATRIX (held-out test, chosen threshold)
# ---------------------------------------------------------------------------
def make_confusion_matrix(model, X_test, test_df, threshold_info):
    threshold = threshold_info["threshold"]
    y_test = test_df["outcome_true"].astype(int)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Predicted LOSE\n(0)", "Predicted WIN\n(1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(["Actual LOSE\n(0)", "Actual WIN\n(1)"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_title(f"Held-out Test Confusion Matrix\n(threshold={threshold:.3f}, cost-optimal)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = f"{FIG_DIR}/confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    return cm


# ---------------------------------------------------------------------------
# 4. ECONOMIC ERROR COST
# ---------------------------------------------------------------------------
def make_economic_error_cost(model, X_test, test_df, threshold_info):
    threshold = threshold_info["threshold"]
    fight_cost_flat = threshold_info.get("fight_cost_flat", 300.0)

    y_test = test_df["outcome_true"].astype(int)
    amounts = test_df["amount"].values
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    fp_mask = (preds == 1) & (y_test.values == 0)  # predicted WIN, actually LOSE
    fn_mask = (preds == 0) & (y_test.values == 1)  # predicted LOSE, actually WIN

    n_fp = int(fp_mask.sum())
    fight_loss_cost = n_fp * fight_cost_flat
    money_left_on_table = float(amounts[fn_mask].sum())
    n_fn = int(fn_mask.sum())
    total_cost = fight_loss_cost + money_left_on_table

    fig, ax = plt.subplots(figsize=(7, 5.5))
    categories = ["Fight-Loss Cost\n(FIGHT but LOST)", "Money Left on Table\n(CONCEDED but WINNABLE)"]
    values = [fight_loss_cost, money_left_on_table]
    counts = [n_fp, n_fn]
    colors = ["#ff7f0e", "#d62728"]

    bars = ax.bar(categories, values, color=colors, width=0.5)
    for bar, val, cnt in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + total_cost * 0.02,
                 f"Rs.{val:,.0f}\n({cnt} cases)", ha="center", va="bottom", fontweight="bold")

    ax.axhline(total_cost, color="gray", linestyle="--", linewidth=1)
    ax.text(1.4, total_cost, f"Total: Rs.{total_cost:,.0f}", va="center", fontsize=10, color="gray")

    ax.set_ylabel("Rs.")
    ax.set_title(f"Economic Error Cost — Held-out Test Set\n(fight cost assumption: Rs.{fight_cost_flat:.0f}/case)")
    ax.set_ylim(0, total_cost * 1.25)
    plt.tight_layout()
    path = f"{FIG_DIR}/economic_error_cost.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    print(f"\nFight-loss cost      : Rs.{fight_loss_cost:,.2f}  ({n_fp} cases)")
    print(f"Money left on table  : Rs.{money_left_on_table:,.2f}  ({n_fn} cases)")
    print(f"Total modeled cost   : Rs.{total_cost:,.2f}")

    # ---------------------------------------------------------------------------
# 5. ROC-AUC CURVE — HELD-OUT TEST SET
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score


def make_roc_auc_curve(model, X_test, test_df):
    y_test = test_df["outcome_true"].astype(int)

    # Predicted probabilities — no threshold applied
    test_probs = model.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    auc_score = roc_auc_score(y_test, test_probs)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        label=f"XGBoost (ROC-AUC = {auc_score:.4f})",
    )

    # Random classifier baseline
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random baseline (AUC = 0.5000)",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(
        "ROC Curve — Held-out Test Set"
    )

    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    path = f"{FIG_DIR}/roc_auc_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print(f"Saved {path}")
    print(f"ROC-AUC: {auc_score:.4f}")


def main():
    model, feature_columns, threshold_info, test_df, X_test = load_common()

    print("Generating 4 core evaluation figures...\n")
    make_shap_beeswarm(model, X_test, feature_columns)
    make_review_margin_sensitivity()
    make_confusion_matrix(model, X_test, test_df, threshold_info)
    make_economic_error_cost(model, X_test, test_df, threshold_info)
    make_roc_auc_curve(
        model,
        X_test,
        test_df
    )

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()